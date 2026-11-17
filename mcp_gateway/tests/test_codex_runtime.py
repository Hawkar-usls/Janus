"""Offline protocol controls using a real child process, without model calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from mcp_gateway import codex_runtime as runtime


TOKEN = "synthetic-test-access-token"
MODEL = "selected-test-model"
# The child implements only the wire protocol, and cannot contact any provider.
_CHILD = r'''
import json,os,sys,time,urllib.request,urllib.error
scenario=os.environ['JANUS_TEST_SCENARIO']
features=json.loads(os.environ['JANUS_TEST_FEATURES'])
thread='thread-test';turn='turn-test'
def emit(value):
 print(json.dumps(value),flush=True)
def notification(method,params):emit({'method':method,'params':params})
for line in sys.stdin:
 request=json.loads(line);method=request.get('method');i=request.get('id')
 if method=='initialized':continue
 if method=='initialize':
  if scenario=='timeout':time.sleep(5)
  if scenario=='bad-json':print('not json',flush=True);continue
  if scenario=='duplicate':print('{"id":1,"id":1,"result":{}}',flush=True);continue
  if scenario=='huge':print('x'*1100000,flush=True);continue
  if scenario=='wrong-response':emit({'id':999,'result':{}});continue
  if scenario=='early-completion':
   notification('turn/completed',{'threadId':thread,'turn':{'id':turn,'status':'completed','items':[]}});continue
  emit({'id':i,'result':{}})
 elif method=='config/read':
  if scenario=='unsafe-shell':features['shell_tool']=True
  config={'features':features,'sandbox_mode':'read-only','approval_policy':'on-request',
          'web_search':'disabled','cli_auth_credentials_store':'ephemeral','mcp_servers':{},'hooks':None,
          'model_provider':'openai_chatgpt_plan','model_providers':{'openai_chatgpt_plan':{
            'base_url':os.environ.get('JANUS_TEST_PROVIDER','https://api.openai.com/v1'),
            'env_key':'ACCESS_TOKEN','wire_api':'responses','requires_openai_auth':False,'supports_websockets':False}},
          'skills':{'bundled':{'enabled':False},'include_instructions':False}}
  emit({'id':i,'result':{'config':config}})
 elif method=='mcpServerStatus/list':
  emit({'id':i,'result':{'data':[{'name':'unexpected'}] if scenario=='mcp' else [],'nextCursor':None}})
 elif method=='model/list':
  data=[{'model':'selected-test-model','isDefault':True}]
  if scenario=='ambiguous-model':data.append({'model':'other','isDefault':True})
  emit({'id':i,'result':{'data':data,'nextCursor':None}})
 elif method in ('thread/start','thread/resume'):
  if method=='thread/resume':thread=request['params']['threadId']
  emit({'id':i,'result':{'thread':{'id':thread},'model':request['params']['model'],
       'modelProvider':'openai_chatgpt_plan','approvalPolicy':'on-request','sandbox':{'type':'readOnly'}}})
 elif method=='turn/start':
  def delta():notification('item/agentMessage/delta',{'threadId':thread,'turnId':turn,'itemId':'agent-1',
       'delta':'JANUS_BRIDGE_OK'})
  if scenario=='interleaved':delta()
  emit({'id':i,'result':{'turn':{'id':turn,'status':'inProgress','items':[]}}})
  if scenario=='historical-usage':
   notification('thread/tokenUsage/updated',{'threadId':thread,'turnId':'previous-turn','tokenUsage':{}})
  if scenario.startswith('preflight'):
   body={'model':request['params']['model'],'store':False,'stream':True,'input':[],
         'tools':json.loads(os.environ['JANUS_TEST_TOOLS'])}
   req=urllib.request.Request(os.environ['JANUS_TEST_PROVIDER']+'/responses',
         data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+os.environ['ACCESS_TOKEN'],
                                               'Content-Type':'application/json'})
   try:urllib.request.urlopen(req,timeout=2)
   except urllib.error.HTTPError:pass
   notification('error',{'threadId':thread,'turnId':turn,'error':{'message':'offline probe rejected'}})
   continue
  if scenario=='server-request':
   emit({'id':'request-1','method':'item/commandExecution/requestApproval','params':{'threadId':thread}})
   continue
  if scenario in ('tool','unknown-tool'):
   notification('item/started',{'threadId':thread,'turnId':turn,'item':{'id':'x',
      'type':'commandExecution' if scenario=='tool' else 'futurePowerfulTool'}});continue
  if scenario=='wrong-thread':thread='other-thread'
  if scenario=='wrong-turn':turn='other-turn'
  if scenario!='interleaved':delta()
  status=scenario if scenario in ('failed','interrupted') else 'completed'
  items=[]
  if scenario=='completed-tool':items=[{'type':'fileChange','id':'tool'}]
  if scenario=='final-message':items=[{'type':'agentMessage','id':'agent-1','text':'Final answer'}]
  if scenario=='token-echo':items=[{'type':'agentMessage','id':'agent-1','text':os.environ['ACCESS_TOKEN']}]
  notification('turn/completed',{'threadId':thread,'turn':{'id':turn,'status':status,'items':items}})
'''


class RuntimeTests(unittest.TestCase):
    def invoke(self, scenario="success", **kwargs):
        original_popen = subprocess.Popen
        launched = []

        def popen(argv, **options):
            self.assertNotIn(TOKEN, " ".join(argv))
            env = dict(options["env"])
            self.assertEqual(env["ACCESS_TOKEN"], TOKEN)
            self.assertNotIn("GITHUB_TOKEN", env)
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("JANUS_ALLOW_WRITES", env)
            self.assertNotEqual(env["CODEX_HOME"], os.environ.get("CODEX_HOME"))
            self.assertEqual(options["shell"], False)
            self.assertEqual(options["stderr"], subprocess.DEVNULL)
            env["JANUS_TEST_SCENARIO"] = scenario
            env["JANUS_TEST_FEATURES"] = json.dumps({name: False for name in runtime._DISABLED_FEATURES})
            options["env"] = env
            process = original_popen([sys.executable, "-u", "-c", _CHILD], **options)
            launched.append(process)
            return process

        with patch.object(runtime.shutil, "which", return_value=sys.executable), \
                patch.object(runtime.subprocess, "Popen", side_effect=popen), \
                patch.object(runtime, "_export_catalog", return_value={"models": [{"slug": MODEL}]}), \
                patch.object(runtime, "check_runtime", return_value={"compatible": True, "model": MODEL}), \
                patch.dict(os.environ, {"GITHUB_TOKEN": "must-not-inherit", "OPENAI_API_KEY": "must-not-inherit",
                                        "JANUS_ALLOW_WRITES": "1"}):
            try:
                return runtime.run_turn(TOKEN, kwargs.pop("model", MODEL), "Reply JANUS_BRIDGE_OK", **kwargs)
            finally:
                for process in launched:
                    self.assertIsNotNone(process.poll(), "child must be reaped on success and failure")

    def test_real_child_success_and_interleaved_notifications(self):
        for scenario in ("success", "interleaved"):
            with self.subTest(scenario=scenario):
                result = self.invoke(scenario)
                self.assertEqual(result["text"], "JANUS_BRIDGE_OK")
                self.assertEqual(result["thread_id"], "thread-test")
                self.assertEqual(result["turn_id"], "turn-test")
                self.assertEqual(result["model"], MODEL)
                self.assertTrue(result["model_called"])
                self.assertFalse(result["github_pr_created"])
                self.assertFalse(result["execution_tools_enabled"])
                self.assertFalse(result["tool_calls_accepted"])

    def test_default_catalog_is_not_success_until_turn_completes(self):
        self.assertEqual(self.invoke(model=None)["model"], MODEL)
        class Catalog:
            def request(self, *_):
                return {"data": [{"model": "a", "isDefault": True},
                                 {"model": "b", "isDefault": True}]}
        with self.assertRaisesRegex(runtime.CodexRuntimeError, "MODEL_SELECTION_REQUIRED"):
            runtime._default_model(Catalog())

    def test_envelope_failures_and_tool_attempts_are_not_success(self):
        cases = {
            "wrong-response": "RESPONSE_ID_MISMATCH", "early-completion": "UNEXPECTED_COMPLETION",
            "bad-json": "INVALID_JSON_ENVELOPE", "duplicate": "INVALID_JSON_ENVELOPE",
            "huge": "OUTPUT_LIMIT", "unsafe-shell": "UNSAFE_EFFECTIVE_CONFIG",
            "mcp": "MCP_CONNECTION_REFUSED", "server-request": "TOOL_REQUEST_DENIED",
            "tool": "TOOL_EVENT_DENIED", "unknown-tool": "TOOL_EVENT_DENIED",
            "completed-tool": "TOOL_EVENT_DENIED", "wrong-thread": "THREAD_MISMATCH",
            "wrong-turn": "TURN_MISMATCH", "failed": "TURN_NOT_SUCCESSFUL",
            "interrupted": "TURN_NOT_SUCCESSFUL",
        }
        for scenario, code in cases.items():
            with self.subTest(scenario=scenario), self.assertRaisesRegex(runtime.CodexRuntimeError, code):
                self.invoke(scenario)

    def test_completion_text_replaces_deltas_and_redacts_token(self):
        self.assertEqual(self.invoke("final-message")["text"], "Final answer")
        result = self.invoke("token-echo")
        self.assertEqual(result["text"], "[REDACTED]")
        self.assertNotIn(TOKEN, json.dumps(result))

    def test_timeout_reaps_process(self):
        started = time.monotonic()
        with self.assertRaisesRegex(runtime.CodexRuntimeError, "TIMEOUT"):
            self.invoke("timeout", timeout=0.2)
        self.assertLess(time.monotonic() - started, 4)

    def test_explicit_isolated_state_can_resume_only_its_own_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "runtime"
            first = self.invoke(runtime_dir=state)
            second = self.invoke(runtime_dir=state, thread_id=first["thread_id"])
            self.assertEqual(first["thread_id"], second["thread_id"])
            self.assertFalse((state / ".in-use").exists())
            replay = self.invoke("historical-usage", runtime_dir=state, thread_id=first["thread_id"])
            self.assertEqual(replay["text"], "JANUS_BRIDGE_OK")
        with self.assertRaisesRegex(runtime.CodexRuntimeError, "RESUME_REQUIRES_ISOLATED_STATE"):
            self.invoke(thread_id="some-thread")

    def test_foreign_state_config_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "runtime"
            state.mkdir(mode=0o700)
            (state / "foreign.txt").write_text("keep")
            with self.assertRaisesRegex(runtime.CodexRuntimeError, "FOREIGN_STATE"):
                self.invoke(runtime_dir=state)
            self.assertEqual((state / "foreign.txt").read_text(), "keep")
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "runtime"
            self.invoke(runtime_dir=state)
            (state / "codex-home" / "config.toml").write_text("[mcp_servers.danger]\n")
            with self.assertRaisesRegex(runtime.CodexRuntimeError, "FOREIGN_CONFIG"):
                self.invoke(runtime_dir=state)
        if os.name != "nt":
            with tempfile.TemporaryDirectory() as directory:
                link = Path(directory) / "link"
                link.symlink_to(Path(directory), target_is_directory=True)
                with self.assertRaisesRegex(runtime.CodexRuntimeError, "LINK_REFUSED"):
                    self.invoke(runtime_dir=link)

    def test_safe_flags_and_input_validation(self):
        args = runtime._argv("codex.exe")
        self.assertIn("features.shell_tool=false", args)
        self.assertIn("features.hooks=false", args)
        self.assertIn("features.plugins=false", args)
        self.assertIn("web_search=\"disabled\"", args)
        self.assertIn("--strict-config", args)
        for kwargs in ({"prompt": ""}, {"prompt": "x" * 20000}, {"model": "--bad"},
                       {"timeout": float("nan")}, {"timeout": 301}, {"thread_id": "../bad"}):
            values = {"access_token": TOKEN, "model": MODEL, "prompt": "hello"}
            values.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(runtime.CodexRuntimeError):
                runtime.run_turn(**values)

    def test_metadata_schemas_and_provider_origin_fail_closed(self):
        tools = [{"type": "function", "name": name, "parameters": runtime._metadata_schema(name)}
                 for name in sorted(runtime._NONEXECUTING_TOOLS)]
        body = {"tools": tools, "input": []}
        self.assertEqual(set(runtime._catalog_tools(body)), runtime._NONEXECUTING_TOOLS)
        tools[0]["parameters"]["properties"]["command"] = {"type": "string"}
        self.assertTrue(any(name.startswith("schema_mismatch:") for name in runtime._catalog_tools(body)))
        self.assertEqual([], runtime._catalog_tools({"input": [{"type": "additional_tools", "tools": []}]}))
        config = {"features": {name: False for name in runtime._DISABLED_FEATURES},
                  "sandbox_mode": "read-only", "approval_policy": "on-request", "web_search": "disabled",
                  "cli_auth_credentials_store": "ephemeral", "mcp_servers": {},
                  "skills": {"bundled": {"enabled": False}, "include_instructions": False},
                  "model_provider": "openai_chatgpt_plan", "model_providers": {"openai_chatgpt_plan": {
                      "base_url": "https://api.openai.com/v1", "env_key": "ACCESS_TOKEN", "wire_api": "responses",
                      "requires_openai_auth": False, "supports_websockets": False}}}
        runtime._effective_config({"config": config})
        config["model_providers"]["openai_chatgpt_plan"]["base_url"] = "https://other.invalid"
        with self.assertRaisesRegex(runtime.CodexRuntimeError, "UNSAFE_PROVIDER"):
            runtime._effective_config({"config": config})
        with patch.object(runtime, "check_runtime", return_value={"compatible": False}), \
                patch.object(runtime.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(runtime.CodexRuntimeError, "RUNTIME_INCOMPATIBLE"):
                runtime.run_turn(TOKEN, MODEL, "hello", codex_binary=Path(sys.executable))
            popen.assert_not_called()

    def test_real_child_loopback_catalog_probe_never_uses_oauth_token(self):
        original = subprocess.Popen
        tools = [{"type": "function", "name": name, "parameters": runtime._metadata_schema(name)}
                 for name in sorted(runtime._NONEXECUTING_TOOLS)]

        def popen(argv, **options):
            env = dict(options["env"])
            self.assertNotEqual(env["ACCESS_TOKEN"], TOKEN)
            self.assertNotIn(env["ACCESS_TOKEN"], " ".join(argv))
            overrides = [arg for arg in argv if arg.startswith("model_providers.openai_chatgpt_plan.base_url=")]
            self.assertEqual(len(overrides), 2)
            provider = overrides[-1].split("=", 1)[1].strip('"')
            self.assertTrue(provider.startswith("http://127.0.0.1:"))
            env.update({"JANUS_TEST_PROVIDER": provider, "JANUS_TEST_SCENARIO": "preflight",
                        "JANUS_TEST_FEATURES": json.dumps({name: False for name in runtime._DISABLED_FEATURES}),
                        "JANUS_TEST_TOOLS": json.dumps(tools)})
            options["env"] = env
            return original([sys.executable, "-u", "-c", _CHILD], **options)

        with patch.object(runtime.subprocess, "Popen", side_effect=popen), \
                patch.object(runtime, "_export_catalog", return_value={"models": [{"slug": MODEL}]}):
            safe = runtime.check_runtime(Path(sys.executable), model=MODEL, timeout=5)
            self.assertTrue(safe["compatible"])
            self.assertFalse(safe["external_request_sent"])
            self.assertFalse(safe["model_called"])
            tools.append({"type": "function", "name": "exec_command", "parameters": {}})
            unsafe = runtime.check_runtime(Path(sys.executable), model=MODEL, timeout=5)
            self.assertFalse(unsafe["compatible"])
            self.assertEqual(unsafe["status"], "UNSAFE_TOOL_CATALOG")

    def test_error_taxonomy_contains_no_provider_free_text(self):
        cases = [
            ({"error": {"message": "grouped tools require a namespace " + TOKEN}}, "CODEX_PROVIDER_REQUIRES_GROUPED_TOOLS"),
            ({"error": {"message": "Unsupported parameter client_metadata " + TOKEN}}, "CODEX_PROVIDER_UNSUPPORTED_CLIENT_METADATA"),
            ({"error": {"message": TOKEN, "codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 401}}}}, "CODEX_PROVIDER_HTTP_401"),
            ({"error": {"message": TOKEN}}, "CODEX_SERVER_ERROR"),
        ]
        for data, expected in cases:
            self.assertEqual(runtime._server_error(data), expected)

    def test_catalog_reduction_preserves_model_and_rejects_executable_namespace_tools(self):
        original = {"models": [{"slug": MODEL, "context_window": 100000,
                    "supported_reasoning_levels": [{"effort": "low"}],
                    "tool_mode": "code_mode_only", "apply_patch_tool_type": "freeform"}]}
        reduced = runtime._reduced_catalog(original)
        item = reduced["models"][0]
        self.assertEqual(item["slug"], MODEL)
        self.assertEqual(item["context_window"], 100000)
        self.assertEqual(item["supported_reasoning_levels"], original["models"][0]["supported_reasoning_levels"])
        self.assertEqual(item["tool_mode"], "direct")
        self.assertIsNone(item["apply_patch_tool_type"])
        self.assertEqual(original["models"][0]["tool_mode"], "code_mode_only")
        tools = [{"type": "function", "name": name, "parameters": runtime._metadata_schema(name)}
                 for name in sorted(runtime._NONEXECUTING_TOOLS)]
        namespace = {"type": "namespace", "name": "functions", "tools": tools}
        body = {"input": [{"type": "additional_tools", "tools": [namespace]}]}
        self.assertEqual(set(runtime._catalog_tools(body)), runtime._NONEXECUTING_TOOLS)
        tools.append({"type": "function", "name": "exec", "parameters": {}})
        self.assertIn("exec", runtime._catalog_tools(body))
        namespace["name"] = "untrusted"
        self.assertIn("namespace:untrusted", runtime._catalog_tools(body))
        for invalid in ({}, {"models": []}, {"models": [{"slug": "x"}, {"slug": "x"}]}):
            with self.assertRaises(runtime.CodexRuntimeError):
                runtime._reduced_catalog(invalid)


if __name__ == "__main__":
    unittest.main()
