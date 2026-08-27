from pathlib import Path
p = Path('firmware/ADV_Elite_RC2_1_RECOVERY/ADV_Elite_RC2_1_RECOVERY.ino')
s = p.read_text(encoding='utf-8')

s = s.replace(
    'uint8_t brightness=170;\nString statusLine="RC2.1 READY";',
    'uint8_t brightness=170;\nuint8_t visualSource=0;\nfloat visualGain=1.0f;\nuint32_t lastCatalogAttempt=0;\nString statusLine="RC2.1 READY";'
)

# M5Cardputer 1.1.1 on Cardputer ADV exposes fn + printable word and del,
# but not the newer master-only esc/left/right/up/down/backspace booleans.
# The physical ADV legends are decoded directly from the Fn layer:
# Fn+` ESC, Fn+, LEFT, Fn+/ RIGHT, Fn+; UP, Fn+. DOWN.
s = s.replace("  bool esc=ks.esc||rawFn(ks,'`')||rawFn(ks,'~');", "  bool esc=rawFn(ks,'`')||rawFn(ks,'~');")
s = s.replace("  bool left=ks.left||rawFn(ks,',');", "  bool left=rawFn(ks,',');")
s = s.replace("  bool right=ks.right||rawFn(ks,'/');", "  bool right=rawFn(ks,'/');")
s = s.replace("  bool up=ks.up||rawFn(ks,';');", "  bool up=rawFn(ks,';');")
s = s.replace("  bool down=ks.down||rawFn(ks,'.');", "  bool down=rawFn(ks,'.');")
s = s.replace("      if(ks.backspace&&wifiTyped.length())wifiTyped.remove(wifiTyped.length()-1);", "      if(ks.del&&wifiTyped.length())wifiTyped.remove(wifiTyped.length()-1);")

s = s.replace(
    '  static uint8_t visSource=0; static float visGain=1.0f;\n  if(mode==Mode::VIS){if(rise(left,prev.l))visSource=(visSource+3)%4;if(rise(right,prev.r))visSource=(visSource+1)%4;if(rise(up,prev.u))visGain=min(4.0f,visGain*1.2f);if(rise(down,prev.d))visGain=max(0.5f,visGain/1.2f);}',
    '  if(mode==Mode::VIS){if(rise(left,prev.l))visualSource=(visualSource+3)%4;if(rise(right,prev.r))visualSource=(visualSource+1)%4;if(rise(up,prev.u))visualGain=min(4.0f,visualGain*1.2f);if(rise(down,prev.d))visualGain=max(0.5f,visualGain/1.2f);}'
)

s = s.replace(
    '  frame.fillScreen(frame.color565(2,4,7));drawHeader("O / OSCILLOSCOPE + KALEIDO");uint16_t p=themePrimary();',
    '  frame.fillScreen(frame.color565(2,4,7)); const char* vn[4]={"ENTROPY","IMU","ENV","KALEIDO"}; char vh[48]; snprintf(vh,sizeof(vh),"O / %s  x%.1f",vn[visualSource],visualGain); drawHeader(vh);uint16_t p=themePrimary();'
)

s = s.replace(
    '  if(visCount>2){for(int i=0;i<visCount-1;i++){int a=(visPos+120-visCount+i)%120,b=(a+1)%120;int x1=6+i*228/max(1,(int)visCount-1),x2=6+(i+1)*228/max(1,(int)visCount-1);int y1=72-(int)(constrain(visHist[a]/2.0f,0.0f,1.0f)*43),y2=72-(int)(constrain(visHist[b]/2.0f,0.0f,1.0f)*43);frame.drawLine(x1,y1,x2,y2,p);}}',
    '  if(visualSource!=3 && visCount>2){float scale=(visualSource==0?2.0f:(visualSource==1?1.4f:3.0f))/visualGain;for(int i=0;i<visCount-1;i++){int a=(visPos+120-visCount+i)%120,b=(a+1)%120;int x1=6+i*228/max(1,(int)visCount-1),x2=6+(i+1)*228/max(1,(int)visCount-1);float va=visHist[a];float vb=visHist[b];if(visualSource==1){va=fabsf(sinf(va*2.7f))*core.shock;vb=fabsf(sinf(vb*2.7f))*core.shock;}else if(visualSource==2){float ep=env.qmpOk?fabsf(env.pressureHpa-1000.0f)/25.0f:0.0f;va=va*0.25f+ep;vb=vb*0.25f+ep;}int y1=72-(int)(constrain(va/scale,0.0f,1.0f)*43),y2=72-(int)(constrain(vb/scale,0.0f,1.0f)*43);frame.drawLine(x1,y1,x2,y2,p);}}'
)

s = s.replace(
    '  int cx=120,cy=89;for(int ring=0;ring<5;ring++){float rr=8+ring*6;for(int k=0;k<8;k++){float a=k*PI/4+millis()*0.0004f*(ring&1?1:-1);frame.drawPixel(cx+cosf(a)*rr,cy+sinf(a)*rr*0.6f,p);}}',
    '  int cx=120,cy=visualSource==3?70:89;int rings=visualSource==3?9:5;for(int ring=0;ring<rings;ring++){float rr=8+ring*(visualSource==3?5.0f:6.0f);for(int k=0;k<8;k++){float a=k*PI/4+millis()*0.0004f*(ring&1?1:-1);int rr2=(int)(rr + sinf(millis()*0.002f+ring)*3.0f*visualGain);frame.drawPixel(cx+cosf(a)*rr2,cy+sinf(a)*rr2*0.6f,p);}}'
)

s = s.replace(
    '  radioTick();brainWaveTick();',
    '  if(mode==Mode::RADIO && WiFi.status()==WL_CONNECTED && stationCount==0 && !radioCatalogBusy && now-lastCatalogAttempt>5000UL){lastCatalogAttempt=now;refreshRadioCatalog();}\n  radioTick();brainWaveTick();'
)

p.write_text(s, encoding='utf-8')
print('patched', p)
