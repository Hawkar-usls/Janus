/*
  JANUS ADV_ELITE RC2.1 RECOVERY
  Target: M5Stack Cardputer / Cardputer ADV
  Goals: Beacon-style double-buffered UI, real Fn arrows, universal ESC/G0 HOME,
         ENV III pressure fix, WiFi setup on-device, ESP8266Audio web radio,
         BrainWave audio ownership, 1488 turquoise / 112269 amber priority,
         smooth lightweight neon survival game.
*/

#include <M5Cardputer.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <Preferences.h>
#include <M5UnitENV.h>
#include <ArduinoJson.h>
#include <AudioOutput.h>
#include <AudioFileSourceICYStream.h>
#include <AudioFileSourceBuffer.h>
#include <AudioGeneratorMP3.h>
#include <math.h>

static constexpr int SW = 240;
static constexpr int SH = 135;
static constexpr uint8_t ENV_SDA = 2;
static constexpr uint8_t ENV_SCL = 1;

M5Canvas frame(&M5Cardputer.Display);
Preferences prefs;
SHT3X sht;
QMP6988 qmp;

struct EnvState {
  volatile bool shtOk = false;
  volatile bool qmpOk = false;
  volatile float tempC = NAN;
  volatile float hum = NAN;
  volatile float pressureHpa = NAN;
  volatile uint32_t stamp = 0;
} env;

struct CoreState {
  float ax=0, ay=0, az=1;
  float gx=0, gy=0, gz=0;
  float shock=1.0f, pred=1.0f, loss=0.0f;
  float entropy=0.20f, predEntropy=0.20f, trend=0.0f;
  int battery=0;
  int rssi=-127;
  uint32_t heap=0;
} core;

enum class Mode : uint8_t { HOME, VIS, SELF, RADIO, PET, GAME, WIFI_LIST, WIFI_PASS };
Mode mode = Mode::HOME;
bool houseActive=false;
bool m2rActive=false;
bool hardMute=false;
bool brainWaveEnabled=true;
uint8_t volumeLevel=104;
uint8_t brightness=170;
String statusLine="RC2.1 READY";
String codeBuffer;

uint16_t themePrimary() {
  if (houseActive) return frame.color565(255, 157, 32);   // 112269 absolute priority
  if (m2rActive)   return frame.color565(25, 235, 220);   // 1488 turquoise
  return frame.color565(65, 205, 255);
}
uint16_t themeSecondary() {
  if (houseActive) return frame.color565(150, 82, 20);
  if (m2rActive)   return frame.color565(15, 115, 115);
  return frame.color565(27, 78, 104);
}
uint16_t themeDim() {
  if (houseActive) return frame.color565(82, 48, 18);
  if (m2rActive)   return frame.color565(10, 58, 61);
  return frame.color565(15, 42, 55);
}

// ---------------- Audio output for ESP8266Audio -> M5Unified speaker ----------------
class AudioOutputM5Speaker : public AudioOutput {
public:
  explicit AudioOutputM5Speaker(m5::Speaker_Class* s, uint8_t ch=1): speaker(s), channel(ch) {}
  bool begin() override { return true; }
  bool ConsumeSample(int16_t sample[2]) override {
    if (idx + 1 < BUF) {
      data[bank][idx++] = sample[0];
      data[bank][idx++] = sample[1];
      return true;
    }
    flush();
    return false;
  }
  void flush() override {
    if (!idx) return;
    speaker->playRaw(data[bank], idx, hertz, true, 1, channel);
    bank = bank < 2 ? bank + 1 : 0;
    idx = 0;
  }
  bool stop() override {
    flush();
    speaker->stop(channel);
    return true;
  }
private:
  m5::Speaker_Class* speaker;
  uint8_t channel;
  static constexpr size_t BUF=1536;
  int16_t data[3][BUF] = {};
  size_t idx=0, bank=0;
};

struct Station {
  char name[42] = {};
  char url[180] = {};
  uint16_t bitrate=0;
};
static constexpr uint8_t MAX_STATIONS=16;
Station stations[MAX_STATIONS];
uint8_t stationCount=0, stationIndex=0;
bool radioDesired=false, radioRunning=false, radioCatalogBusy=false;
uint32_t radioFailures=0;
AudioFileSourceICYStream* radioFile=nullptr;
AudioFileSourceBuffer* radioBuf=nullptr;
AudioGeneratorMP3* radioMp3=nullptr;
AudioOutputM5Speaker* radioOut=nullptr;

void radioStop() {
  if (radioMp3) { radioMp3->stop(); delete radioMp3; radioMp3=nullptr; }
  if (radioBuf) { delete radioBuf; radioBuf=nullptr; }
  if (radioFile) { radioFile->close(); delete radioFile; radioFile=nullptr; }
  if (radioOut) { radioOut->stop(); delete radioOut; radioOut=nullptr; }
  radioRunning=false;
}

bool radioStart() {
  if (hardMute || WiFi.status()!=WL_CONNECTED || stationCount==0) return false;
  radioStop();
  M5Cardputer.Speaker.begin();
  M5Cardputer.Speaker.setVolume(volumeLevel);
  radioFile = new AudioFileSourceICYStream(stations[stationIndex].url);
  radioFile->SetReconnect(3, 250);
  radioBuf = new AudioFileSourceBuffer(radioFile, 8192);
  radioOut = new AudioOutputM5Speaker(&M5Cardputer.Speaker, 1);
  radioMp3 = new AudioGeneratorMP3();
  radioRunning = radioMp3->begin(radioBuf, radioOut);
  if (!radioRunning) { ++radioFailures; radioStop(); }
  return radioRunning;
}

void radioTick() {
  if (mode != Mode::RADIO || hardMute || !radioDesired) {
    if (radioRunning) radioStop();
    return;
  }
  if (!radioRunning) { radioStart(); return; }
  if (!radioMp3->loop()) { ++radioFailures; radioStop(); }
}

bool refreshRadioCatalog() {
  if (WiFi.status()!=WL_CONNECTED) return false;
  radioCatalogBusy=true;
  WiFiClientSecure c; c.setInsecure();
  HTTPClient h;
  const char* url="https://de1.api.radio-browser.info/json/stations/search?codec=MP3&is_https=false&hidebroken=true&order=votes&reverse=true&limit=24";
  h.setTimeout(5500);
  if (!h.begin(c,url)) { radioCatalogBusy=false; return false; }
  h.addHeader("User-Agent","JANUS-ADV-Elite-RC2.1");
  int code=h.GET();
  if (code!=200) { h.end(); radioCatalogBusy=false; return false; }
  DynamicJsonDocument doc(32768);
  auto err=deserializeJson(doc,h.getStream());
  h.end();
  if (err) { radioCatalogBusy=false; return false; }
  stationCount=0;
  for (JsonObject o : doc.as<JsonArray>()) {
    if (stationCount>=MAX_STATIONS) break;
    const char* u=o["url_resolved"]|"";
    const char* codec=o["codec"]|"";
    if (strncmp(u,"http://",7)!=0 || strcasecmp(codec,"MP3")!=0) continue;
    Station& s=stations[stationCount++];
    strlcpy(s.name,o["name"]|"station",sizeof(s.name));
    strlcpy(s.url,u,sizeof(s.url));
    s.bitrate=o["bitrate"]|0;
  }
  stationIndex=0;
  radioCatalogBusy=false;
  return stationCount>0;
}

// ---------------- WiFi setup on device ----------------
String savedSsid, savedPass;
String wifiNames[10];
int wifiCount=0, wifiPick=0;
String wifiChosen, wifiTyped;
uint32_t wifiBeginMs=0;

void loadNetwork() {
  prefs.begin("adv_net",true);
  savedSsid=prefs.getString("ssid","");
  savedPass=prefs.getString("pass","");
  prefs.end();
  if (savedSsid.length()) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(savedSsid.c_str(),savedPass.c_str());
    wifiBeginMs=millis();
  }
}
void saveNetwork(const String& ssid,const String& pass) {
  prefs.begin("adv_net",false);
  prefs.putString("ssid",ssid); prefs.putString("pass",pass); prefs.end();
  savedSsid=ssid; savedPass=pass;
  WiFi.disconnect(true); delay(40); WiFi.mode(WIFI_STA);
  WiFi.begin(savedSsid.c_str(),savedPass.c_str()); wifiBeginMs=millis();
}
void scanNetworks() {
  radioStop(); radioDesired=false;
  mode=Mode::WIFI_LIST; statusLine="WIFI SCAN";
  WiFi.mode(WIFI_STA); WiFi.disconnect(false,false); delay(50);
  int n=WiFi.scanNetworks(false,true);
  wifiCount=min(n,10); wifiPick=0;
  for (int i=0;i<wifiCount;i++) wifiNames[i]=WiFi.SSID(i);
  WiFi.scanDelete();
}

// ---------------- Environment on background task ----------------
void envTask(void*) {
  Wire.begin(ENV_SDA,ENV_SCL,400000U);
  bool sOk=sht.begin(&Wire,SHT3X_I2C_ADDR,ENV_SDA,ENV_SCL,400000U);
  bool qOk=qmp.begin(&Wire,QMP6988_SLAVE_ADDRESS_L,ENV_SDA,ENV_SCL,400000U);
  if (!qOk) qOk=qmp.begin(&Wire,QMP6988_SLAVE_ADDRESS_H,ENV_SDA,ENV_SCL,400000U);
  env.shtOk=sOk; env.qmpOk=qOk;
  for (;;) {
    if (sOk && sht.update()) { env.tempC=sht.cTemp; env.hum=sht.humidity; env.shtOk=true; }
    if (qOk && qmp.update()) { env.pressureHpa=qmp.pressure/100.0f; env.qmpOk=true; }
    env.stamp=millis();
    vTaskDelay(pdMS_TO_TICKS(1750));
  }
}

// ---------------- Core telemetry ----------------
void updateCore() {
  float ax=0,ay=0,az=1,gx=0,gy=0,gz=0;
  M5.Imu.getAccelData(&ax,&ay,&az);
  M5.Imu.getGyroData(&gx,&gy,&gz);
  core.ax=ax; core.ay=ay; core.az=az; core.gx=gx; core.gy=gy; core.gz=gz;
  float mag=sqrtf(ax*ax+ay*ay+az*az)+sqrtf(gx*gx+gy*gy+gz*gz)*0.008f;
  core.shock=mag; core.loss=fabsf(mag-core.pred); core.pred=core.pred*0.96f+mag*0.04f;
  float envTerm=0;
  if (env.shtOk) envTerm=fabsf(env.tempC-23.0f)*0.025f+fabsf(env.hum-50.0f)*0.004f;
  float prev=core.entropy;
  core.entropy=constrain(0.05f+envTerm+core.loss*0.7f,0.01f,9.0f);
  core.trend=core.trend*0.87f+(core.entropy-prev)*0.13f;
  core.predEntropy=core.predEntropy*0.90f+core.entropy*0.10f;
  core.battery=M5Cardputer.Power.getBatteryLevel();
  core.rssi=WiFi.status()==WL_CONNECTED?WiFi.RSSI():-127;
  core.heap=ESP.getFreeHeap();
}

// ---------------- Input ----------------
struct Edge { bool esc=false,l=false,r=false,u=false,d=false,space=false,enter=false,o=false,z=false,rr=false,dd=false,a=false,n=false,i=false,lb=false,rb=false; } prev;
bool rise(bool now,bool& old){bool x=now&&!old;old=now;return x;}
bool wordHas(const Keyboard_Class::KeysState& ks,char c){for(char w:ks.word)if(w==c)return true;return false;}

bool rawFn(const Keyboard_Class::KeysState& ks,char c){return ks.fn && wordHas(ks,c);}

void goHome() {
  if (mode==Mode::RADIO) { radioDesired=false; radioStop(); }
  mode=Mode::HOME; statusLine="HOME";
}

void handleCodes(const Keyboard_Class::KeysState& ks) {
  if (mode==Mode::WIFI_PASS || mode==Mode::WIFI_LIST) return;
  if (!M5Cardputer.Keyboard.isChange() || !M5Cardputer.Keyboard.isPressed()) return;
  for(char c:ks.word) if(c>='0'&&c<='9') {
    codeBuffer += c; if(codeBuffer.length()>12) codeBuffer.remove(0,codeBuffer.length()-12);
    if(codeBuffer.endsWith("1488")) { m2rActive=!m2rActive; codeBuffer=""; statusLine=m2rActive?"1488 M2R / TURQUOISE":"1488 OFF"; }
    else if(codeBuffer.endsWith("112269")) { houseActive=!houseActive; codeBuffer=""; statusLine=houseActive?"112269 HOUSE / AMBER":"HOUSE OFF"; }
  }
}

// ---------------- Game ----------------
struct Enemy { bool on=false; float x=0,y=0,vx=0,vy=0,hp=1; uint8_t kind=0; } enemies[14];
struct Bullet { bool on=false; float x=0,y=0,vx=0,vy=0; } bullets[18];
struct Spark { bool on=false; float x=0,y=0,vx=0,vy=0,life=0; } sparks[36];
float playerX=120,playerY=76,gameHp=100;
uint32_t gameKills=0,gameWave=1,lastSpawn=0,lastShot=0,lastGameMs=0;
bool autoFire=true;

void gameReset() {
  for(auto& e:enemies)e.on=false; for(auto& b:bullets)b.on=false; for(auto& s:sparks)s.on=false;
  playerX=120;playerY=76;gameHp=100;gameKills=0;gameWave=1;lastSpawn=lastShot=0;lastGameMs=millis();
}
void spawnSpark(float x,float y,uint8_t n=6) {
  while(n--) for(auto& s:sparks) if(!s.on){float a=random(0,6283)*0.001f;float sp=random(15,65);s={true,x,y,cosf(a)*sp,sinf(a)*sp,0.30f};break;}
}
void spawnEnemy() {
  for(auto& e:enemies) if(!e.on) {
    e.on=true;e.kind=random(0,3);e.hp=e.kind==2?3:1;
    int side=random(0,4); if(side==0){e.x=2;e.y=random(18,128);}else if(side==1){e.x=238;e.y=random(18,128);}else if(side==2){e.x=random(4,236);e.y=16;}else{e.x=random(4,236);e.y=130;}
    e.vx=e.vy=0; return;
  }
}
void shootNearest() {
  int idx=-1;float best=1e9;
  for(int i=0;i<14;i++)if(enemies[i].on){float dx=enemies[i].x-playerX,dy=enemies[i].y-playerY,d=dx*dx+dy*dy;if(d<best){best=d;idx=i;}}
  if(idx<0)return;
  for(auto& b:bullets)if(!b.on){float dx=enemies[idx].x-playerX,dy=enemies[idx].y-playerY,l=sqrtf(dx*dx+dy*dy)+0.001f;b={true,playerX,playerY,dx/l*150.0f,dy/l*150.0f};lastShot=millis();return;}
}
void gameTick(bool left,bool right,bool up,bool down,bool fire) {
  uint32_t now=millis();float dt=min(0.05f,(now-lastGameMs)*0.001f);lastGameMs=now;
  float mx=(right?1:0)-(left?1:0),my=(down?1:0)-(up?1:0);float ml=sqrtf(mx*mx+my*my);if(ml>0){mx/=ml;my/=ml;}
  playerX=constrain(playerX+mx*92*dt,8.0f,232.0f);playerY=constrain(playerY+my*92*dt,19.0f,126.0f);
  uint32_t spawnGap=max(230,(int)(720-gameWave*28));if(now-lastSpawn>(uint32_t)spawnGap){lastSpawn=now;spawnEnemy();}
  if((fire||autoFire)&&now-lastShot>170)shootNearest();
  for(auto& b:bullets)if(b.on){b.x+=b.vx*dt;b.y+=b.vy*dt;if(b.x<0||b.x>240||b.y<12||b.y>135)b.on=false;}
  for(auto& e:enemies)if(e.on){float dx=playerX-e.x,dy=playerY-e.y,l=sqrtf(dx*dx+dy*dy)+0.001f;float sp=e.kind==1?48:(e.kind==2?22:32);e.x+=dx/l*sp*dt;e.y+=dy/l*sp*dt;if(l<7){gameHp-=22*dt;if(gameHp<0)gameHp=0;}}
  for(auto& b:bullets)if(b.on)for(auto& e:enemies)if(e.on){float dx=b.x-e.x,dy=b.y-e.y;if(dx*dx+dy*dy<30){b.on=false;e.hp-=1;spawnSpark(e.x,e.y);if(e.hp<=0){e.on=false;gameKills++;if(gameKills%12==0)gameWave++;}break;}}
  for(auto& s:sparks)if(s.on){s.x+=s.vx*dt;s.y+=s.vy*dt;s.life-=dt;if(s.life<=0)s.on=false;}
  if(gameHp<=0 && fire) gameReset();
}

// ---------------- Pet ----------------
float petMood=82,petEnergy=78,petHunger=20;
uint8_t petAction=0; const char* petActions[]={"FEED","PLAY","REST"};
void petAct(){if(petAction==0){petHunger=max(0.0f,petHunger-25);petMood=min(100.0f,petMood+5);}else if(petAction==1){petMood=min(100.0f,petMood+14);petEnergy=max(0.0f,petEnergy-8);}else{petEnergy=min(100.0f,petEnergy+20);}statusLine="PET ACTION";}

// ---------------- BrainWave ----------------
uint32_t lastBrain=0;uint8_t brainStep=0;
const uint16_t brainNotes[8]={220,277,330,440,392,330,277,247};
void brainWaveTick() {
  if(!brainWaveEnabled||hardMute||volumeLevel==0||mode==Mode::RADIO||mode==Mode::GAME||mode==Mode::WIFI_LIST||mode==Mode::WIFI_PASS)return;
  uint32_t now=millis();uint32_t gap=(uint32_t)constrain(390-core.entropy*22,135.0f,390.0f);if(now-lastBrain<gap)return;lastBrain=now;
  uint16_t n=brainNotes[brainStep++&7];if(houseActive)n=(uint16_t)(n*0.89f);if(m2rActive)n=(uint16_t)(n*1.07f);M5Cardputer.Speaker.tone(n,gap/2);
}

void setMute(bool m){hardMute=m;if(m){radioStop();M5Cardputer.Speaker.stop();}M5Cardputer.Speaker.setVolume(m?0:volumeLevel);}

void processInput() {
  M5Cardputer.update();
  auto ks=M5Cardputer.Keyboard.keysState();

  // Modern library gives true special-key states. Raw Fn mappings are a fallback.
  bool esc=ks.esc||rawFn(ks,'`')||rawFn(ks,'~');
  bool left=ks.left||rawFn(ks,',');
  bool right=ks.right||rawFn(ks,'/');
  bool up=ks.up||rawFn(ks,';');
  bool down=ks.down||rawFn(ks,'.');
  bool space=ks.space;

  // Independent physical emergency HOME key on the top edge.
  if (M5Cardputer.BtnA.wasPressed() || rise(esc,prev.esc)) { goHome(); return; }

  if (mode==Mode::WIFI_PASS) {
    if(M5Cardputer.Keyboard.isChange()&&M5Cardputer.Keyboard.isPressed()){
      if(ks.backspace&&wifiTyped.length())wifiTyped.remove(wifiTyped.length()-1);
      for(char c:ks.word)if(c>=32&&c<=126)wifiTyped+=c;
      if(ks.enter){saveNetwork(wifiChosen,wifiTyped);mode=Mode::RADIO;statusLine="WIFI CONNECTING";}
    }
    prev.l=left;prev.r=right;prev.u=up;prev.d=down;prev.space=space;return;
  }
  if (mode==Mode::WIFI_LIST) {
    if(rise(up,prev.u)&&wifiCount)wifiPick=(wifiPick+wifiCount-1)%wifiCount;
    if(rise(down,prev.d)&&wifiCount)wifiPick=(wifiPick+1)%wifiCount;
    if((rise(space,prev.space)||ks.enter)&&wifiCount){wifiChosen=wifiNames[wifiPick];wifiTyped="";mode=Mode::WIFI_PASS;}
    return;
  }

  handleCodes(ks);

  if(rise(ks.enter,prev.enter)){setMute(!hardMute);statusLine=hardMute?"MASTER MUTE":"AUDIO ON";}
  bool lb=wordHas(ks,'['),rb=wordHas(ks,']');
  if(rise(lb,prev.lb)){brightness=brightness>24?brightness-20:8;M5Cardputer.Display.setBrightness(brightness);}
  if(rise(rb,prev.rb)){brightness=brightness<235?brightness+20:255;M5Cardputer.Display.setBrightness(brightness);}

  bool kO=wordHas(ks,'o')||wordHas(ks,'O');bool kZ=wordHas(ks,'z')||wordHas(ks,'Z');bool kR=wordHas(ks,'r')||wordHas(ks,'R');bool kD=wordHas(ks,'d')||wordHas(ks,'D');bool kA=wordHas(ks,'a')||wordHas(ks,'A');bool kN=wordHas(ks,'n')||wordHas(ks,'N');bool kI=wordHas(ks,'i')||wordHas(ks,'I');

  if(mode==Mode::HOME){
    if(rise(kO,prev.o)){mode=Mode::VIS;statusLine="VISUALIZER";}
    if(rise(kZ,prev.z)){mode=Mode::SELF;statusLine="SELF";}
    if(rise(kR,prev.rr)){mode=Mode::RADIO;statusLine="RADIO";M5Cardputer.Speaker.stop();if(WiFi.status()!=WL_CONNECTED&&!savedSsid.length())scanNetworks();else if(WiFi.status()==WL_CONNECTED&&!stationCount)refreshRadioCatalog();}
    if(rise(kD,prev.dd)){mode=Mode::PET;statusLine="PET";}
    if(rise(kA,prev.a)){mode=Mode::GAME;statusLine="NEON SURVIVAL";M5Cardputer.Speaker.stop();gameReset();}
    if(rise(kN,prev.n))scanNetworks();
  } else { prev.o=kO;prev.z=kZ;prev.rr=kR;prev.dd=kD;prev.a=kA;prev.n=kN; }

  static uint8_t visSource=0; static float visGain=1.0f;
  if(mode==Mode::VIS){if(rise(left,prev.l))visSource=(visSource+3)%4;if(rise(right,prev.r))visSource=(visSource+1)%4;if(rise(up,prev.u))visGain=min(4.0f,visGain*1.2f);if(rise(down,prev.d))visGain=max(0.5f,visGain/1.2f);}
  else if(mode==Mode::RADIO){if(rise(left,prev.l)&&stationCount){stationIndex=(stationIndex+stationCount-1)%stationCount;if(radioDesired)radioStart();}if(rise(right,prev.r)&&stationCount){stationIndex=(stationIndex+1)%stationCount;if(radioDesired)radioStart();}if(rise(space,prev.space)){radioDesired=!radioDesired;if(!radioDesired)radioStop();else radioStart();}if(rise(kN,prev.n))scanNetworks();if(rise(kR,prev.rr)&&WiFi.status()==WL_CONNECTED){radioStop();refreshRadioCatalog();}}
  else if(mode==Mode::PET){if(rise(left,prev.l))petAction=(petAction+2)%3;if(rise(right,prev.r))petAction=(petAction+1)%3;if(rise(space,prev.space))petAct();}
  else if(mode==Mode::GAME){if(rise(kI,prev.i))autoFire=!autoFire;gameTick(left,right,up,down,space);}
  else {prev.l=left;prev.r=right;prev.u=up;prev.d=down;prev.space=space;}
}

// ---------------- Rendering helpers ----------------
void chip(int x,int y,int w,const char* label,bool on,uint16_t c){frame.drawRoundRect(x,y,w,12,3,on?c:themeDim());frame.setTextColor(on?c:frame.color565(80,90,95));frame.setCursor(x+4,y+3);frame.print(label);}
void metric(int x,int y,const char* k,const String& v,uint16_t c){frame.setTextColor(frame.color565(115,125,132));frame.setCursor(x,y);frame.print(k);frame.setTextColor(c);frame.setCursor(x+32,y);frame.print(v);}

void drawHeader(const char* title){uint16_t p=themePrimary();frame.fillRect(0,0,SW,14,frame.color565(3,7,10));frame.drawFastHLine(0,13,SW,themeDim());frame.setTextColor(p);frame.setCursor(4,3);frame.print(title);frame.setTextColor(frame.color565(120,130,135));frame.setCursor(193,3);frame.printf("B%02d",core.battery);}

void drawHome(){
  frame.fillScreen(frame.color565(2,5,8));drawHeader("JANUS ADV_ELITE");uint16_t p=themePrimary(),s=themeSecondary();
  frame.drawRoundRect(4,18,112,79,4,themeDim());frame.drawRoundRect(123,18,113,79,4,themeDim());
  frame.setTextColor(s);frame.setCursor(9,22);frame.print("WORLD / SENSORS");frame.setCursor(128,22);frame.print("CORTEX / SELF");
  metric(9,36,"TEMP",env.shtOk?String(env.tempC,1)+"C":"--",p);metric(9,48,"HUM",env.shtOk?String(env.hum,0)+"%":"--",p);metric(9,60,"PRES",env.qmpOk?String(env.pressureHpa,0):"--",p);metric(9,72,"IMU",String(core.shock,2),p);metric(9,84,"WIFI",WiFi.status()==WL_CONNECTED?String(core.rssi):"OFF",p);
  metric(128,36,"ENT",String(core.entropy,3),p);metric(128,48,"PRED",String(core.predEntropy,3),p);metric(128,60,"LOSS",String(core.loss,3),p);metric(128,72,"HEAP",String(core.heap/1024)+"K",p);metric(128,84,"FUT",String(constrain(core.predEntropy+core.trend*3,0.0f,9.0f),2),p);
  chip(4,101,50,"112269",houseActive,p);chip(58,101,42,"1488",m2rActive,p);chip(104,101,38,"ENV",env.shtOk||env.qmpOk,p);chip(146,101,42,"RAD",radioRunning,p);chip(192,101,44,"AUDIO",!hardMute,p);
  frame.setTextColor(frame.color565(115,125,130));frame.setCursor(5,118);frame.print("O VIS  Z SELF  R RADIO  D PET  A GAME");
  frame.setTextColor(s);frame.setCursor(5,128);frame.print(statusLine.substring(0,38));
}

float visHist[120]={};uint16_t visPos=0,visCount=0;uint8_t visSourceGlobal=0;
void sampleVis(){visHist[visPos]=core.entropy;visPos=(visPos+1)%120;if(visCount<120)visCount++;}
void drawVis(){
  frame.fillScreen(frame.color565(2,4,7));drawHeader("O / OSCILLOSCOPE + KALEIDO");uint16_t p=themePrimary();
  auto ks=M5Cardputer.Keyboard.keysState();(void)ks;
  // source page changes are shown through a time-varying hybrid view; navigation itself is handled by real Fn arrows.
  frame.drawRoundRect(4,18,232,101,4,themeDim());
  if(visCount>2){for(int i=0;i<visCount-1;i++){int a=(visPos+120-visCount+i)%120,b=(a+1)%120;int x1=6+i*228/max(1,(int)visCount-1),x2=6+(i+1)*228/max(1,(int)visCount-1);int y1=72-(int)(constrain(visHist[a]/2.0f,0.0f,1.0f)*43),y2=72-(int)(constrain(visHist[b]/2.0f,0.0f,1.0f)*43);frame.drawLine(x1,y1,x2,y2,p);}}
  int cx=120,cy=89;for(int ring=0;ring<5;ring++){float rr=8+ring*6;for(int k=0;k<8;k++){float a=k*PI/4+millis()*0.0004f*(ring&1?1:-1);frame.drawPixel(cx+cosf(a)*rr,cy+sinf(a)*rr*0.6f,p);}}
  frame.setTextColor(frame.color565(110,120,128));frame.setCursor(6,122);frame.print("Fn arrows: L/R source  U/D gain   G0/ESC HOME");
}

void drawSelf(){frame.fillScreen(frame.color565(2,5,8));drawHeader("Z / RESOURCE VIEW");uint16_t p=themePrimary();metric(8,24,"HEAP",String(core.heap),p);metric(8,38,"RSSI",String(core.rssi),p);metric(8,52,"ENV",env.shtOk||env.qmpOk?"ONLINE":"STALE",p);metric(8,66,"RAD",radioRunning?"RUN":"IDLE",p);metric(8,80,"M2R",m2rActive?"ACTIVE":"OFF",p);metric(8,94,"HOUSE",houseActive?"AMBER":"OFF",p);frame.setTextColor(frame.color565(115,125,130));frame.setCursor(8,121);frame.print("G0 or Fn+ESC -> HOME");}

void drawRadio(){
  frame.fillScreen(frame.color565(2,5,8));drawHeader("R / INTERNET RADIO");uint16_t p=themePrimary();
  if(WiFi.status()!=WL_CONNECTED){frame.setTextColor(frame.color565(255,190,70));frame.setCursor(8,28);frame.print(savedSsid.length()?"Connecting WiFi...":"No WiFi configured");frame.setTextColor(p);frame.setCursor(8,46);frame.print("Press N for WiFi setup");}
  else if(radioCatalogBusy){frame.setTextColor(p);frame.setCursor(8,32);frame.print("Loading Radio Browser...");}
  else if(!stationCount){frame.setTextColor(frame.color565(255,160,80));frame.setCursor(8,28);frame.print("No MP3 stations loaded");frame.setTextColor(p);frame.setCursor(8,46);frame.print("Press R to refresh");}
  else {Station& s=stations[stationIndex];frame.setTextColor(p);frame.setCursor(8,25);frame.printf("%02u/%02u",stationIndex+1,stationCount);frame.setTextColor(frame.color565(225,230,232));frame.setCursor(8,39);frame.print(String(s.name).substring(0,36));frame.setTextColor(frame.color565(115,125,130));frame.setCursor(8,55);frame.printf("MP3 %uk   RSSI %d",s.bitrate,core.rssi);frame.setTextColor(radioRunning?frame.color565(80,255,140):frame.color565(255,190,70));frame.setCursor(8,73);frame.printf("%s   failures %lu",radioRunning?"PLAYING":"PAUSED",(unsigned long)radioFailures);frame.setTextColor(frame.color565(125,135,140));frame.setCursor(8,94);frame.print("Fn <- -> station     SPACE play/pause");frame.setCursor(8,108);frame.print("R refresh   N WiFi   G0/ESC HOME");}
  frame.setTextColor(themeSecondary());frame.setCursor(8,124);frame.print("BrainWave: SILENT while RADIO is foreground");
}

void drawWifi(){
  frame.fillScreen(frame.color565(2,5,8));drawHeader(mode==Mode::WIFI_LIST?"WIFI / SELECT":"WIFI / PASSWORD");uint16_t p=themePrimary();
  if(mode==Mode::WIFI_LIST){if(!wifiCount){frame.setTextColor(frame.color565(255,170,80));frame.setCursor(8,28);frame.print("No networks found. G0 HOME");}else for(int i=0;i<wifiCount&&i<8;i++){int y=21+i*13;bool sel=i==wifiPick;if(sel)frame.fillRoundRect(5,y-2,230,12,3,themeDim());frame.setTextColor(sel?p:frame.color565(150,160,165));frame.setCursor(9,y);frame.print(String(wifiNames[i]).substring(0,34));}frame.setTextColor(frame.color565(110,120,125));frame.setCursor(8,124);frame.print("Fn UP/DOWN  SPACE select  G0/ESC cancel");}
  else {frame.setTextColor(frame.color565(150,160,165));frame.setCursor(8,28);frame.print("SSID:");frame.setTextColor(p);frame.setCursor(45,28);frame.print(wifiChosen.substring(0,27));frame.setTextColor(frame.color565(150,160,165));frame.setCursor(8,49);frame.print("PASS:");frame.setTextColor(p);frame.setCursor(45,49);for(size_t i=0;i<wifiTyped.length()&&i<28;i++)frame.print('*');frame.setTextColor(frame.color565(110,120,125));frame.setCursor(8,76);frame.print("Type password on Cardputer keyboard");frame.setCursor(8,90);frame.print("ENTER save/connect   BS delete");frame.setCursor(8,124);frame.print("G0/ESC cancel");}
}

void drawPet(){frame.fillScreen(frame.color565(2,5,8));drawHeader("D / JANUS PET");uint16_t p=themePrimary();frame.setTextColor(p);frame.setCursor(92,27);frame.print("/\\_/\\");frame.setCursor(90,39);frame.print("( o.o )");frame.setCursor(96,51);frame.print("> ^ <");metric(14,72,"MOOD",String((int)petMood),p);metric(14,84,"ENER",String((int)petEnergy),p);metric(128,72,"HUNG",String((int)petHunger),p);frame.setTextColor(frame.color565(245,210,95));frame.setCursor(76,103);frame.printf("< %s >",petActions[petAction]);frame.setTextColor(frame.color565(110,120,125));frame.setCursor(8,124);frame.print("Fn L/R action  SPACE act  G0/ESC HOME");}

void drawGame(){
  frame.fillScreen(frame.color565(1,3,7));uint16_t p=themePrimary();
  for(int i=0;i<34;i++){uint32_t q=(i*1103515245u+gameWave*97u);int x=(q+millis()/35*(1+i%3))%240,y=15+((q>>9)%118);frame.drawPixel(x,y,frame.color565(25+i%25,35+i%35,55+i%50));}
  for(auto& s:sparks)if(s.on)frame.drawPixel((int)s.x,(int)s.y,frame.color565(255,180,80));
  for(auto& b:bullets)if(b.on){frame.fillCircle((int)b.x,(int)b.y,1,frame.color565(110,255,235));}
  for(auto& e:enemies)if(e.on){uint16_t c=e.kind==1?frame.color565(255,80,180):(e.kind==2?frame.color565(255,190,55):frame.color565(90,185,255));int x=e.x,y=e.y;if(e.kind==2){frame.fillRect(x-3,y-3,7,7,c);frame.drawRect(x-5,y-5,11,11,c);}else if(e.kind==1){frame.fillTriangle(x,y-4,x-4,y+4,x+4,y+4,c);}else{frame.drawCircle(x,y,4,c);frame.drawLine(x-5,y,x+5,y,c);}}
  int x=playerX,y=playerY;frame.fillTriangle(x,y-6,x-5,y+5,x+5,y+5,p);frame.drawCircle(x,y,8,themeSecondary());
  frame.fillRect(0,0,240,14,frame.color565(2,6,10));frame.setTextColor(p);frame.setCursor(4,3);frame.printf("A / NEON SURVIVAL  W%lu K%lu",(unsigned long)gameWave,(unsigned long)gameKills);frame.setTextColor(frame.color565(140,150,155));frame.setCursor(188,3);frame.print(autoFire?"AUTO":"MAN");
  frame.drawRect(5,125,82,5,frame.color565(55,60,65));int hp=(int)(80*constrain(gameHp/100.0f,0.0f,1.0f));if(hp)frame.fillRect(6,126,hp,3,gameHp>35?frame.color565(70,240,120):frame.color565(255,70,65));
  if(gameHp<=0){frame.fillRoundRect(55,48,130,38,5,frame.color565(3,4,8));frame.drawRoundRect(55,48,130,38,5,frame.color565(255,70,80));frame.setTextColor(frame.color565(255,90,100));frame.setCursor(92,59);frame.print("SYSTEM DOWN");frame.setTextColor(frame.color565(220,225,228));frame.setCursor(72,73);frame.print("SPACE restart");}
  frame.setTextColor(frame.color565(90,100,108));frame.setCursor(113,126);frame.print("Fn ARROWS move  I auto  G0 HOME");
}

void render(){
  if(mode==Mode::HOME)drawHome();else if(mode==Mode::VIS)drawVis();else if(mode==Mode::SELF)drawSelf();else if(mode==Mode::RADIO)drawRadio();else if(mode==Mode::PET)drawPet();else if(mode==Mode::GAME)drawGame();else drawWifi();
  frame.pushSprite(0,0);
}

void setup(){
  auto cfg=M5.config();M5Cardputer.begin(cfg,true);Serial.begin(115200);
  M5Cardputer.Display.setRotation(1);M5Cardputer.Display.setBrightness(brightness);
  frame.setColorDepth(16);frame.createSprite(SW,SH);frame.setTextSize(1);frame.setTextWrap(false);
  M5Cardputer.Speaker.begin();M5Cardputer.Speaker.setVolume(volumeLevel);
  M5.Imu.init();
  loadNetwork();
  xTaskCreatePinnedToCore(envTask,"env",4096,nullptr,1,nullptr,0);
  gameReset();
  statusLine="RC2.1 RECOVERY READY";
}

void loop(){
  static uint32_t lastCore=0,lastDraw=0,lastVis=0;
  processInput();uint32_t now=millis();
  if(now-lastCore>=80){lastCore=now;updateCore();}
  if(now-lastVis>=120){lastVis=now;sampleVis();}
  radioTick();brainWaveTick();
  uint32_t frameGap=(mode==Mode::GAME)?25:45;
  if(now-lastDraw>=frameGap){lastDraw=now;render();}
  delay(1);
}
