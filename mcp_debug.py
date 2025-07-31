#!/usr/bin/env python3
"""
MCP Server Debug Script

Bu script MCP server'ın stdio modunda düzgün çalışıp çalışmadığını test eder.
"""

import asyncio
import json
import subprocess
import sys
import time
import threading
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_mcp_stdio():
    """Test MCP server in stdio mode like a real MCP client would."""
    
    print("🔍 MCP Server Stdio Mode Debug Test")
    print("=" * 50)
    
    # Start the MCP server process
    print("Starting MCP server process...")
    try:
        process = subprocess.Popen(
            [sys.executable, "mcp_main.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            cwd="."
        )
        print(f"✅ Process started with PID: {process.pid}")
    except Exception as e:
        print(f"❌ Failed to start process: {e}")
        return False
    
    # Give the server a moment to start
    time.sleep(2)
    
    # Check if process is still running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        print(f"❌ Process exited early with code: {process.returncode}")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return False
    
    print("✅ Process is running")
    
    # Test basic communication
    try:
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "debug-client",
                    "version": "1.0.0"
                }
            }
        }
        
        print("\n📤 Sending initialize request...")
        request_json = json.dumps(init_request) + "\n"
        print(f"Request: {request_json.strip()}")
        
        process.stdin.write(request_json)
        process.stdin.flush()
        
        # Read response with timeout
        print("📥 Waiting for response...")
        
        def read_with_timeout(process, timeout=10):
            """Read from process stdout with timeout."""
            result = []
            
            def target():
                try:
                    line = process.stdout.readline()
                    result.append(line)
                except Exception as e:
                    result.append(f"ERROR: {e}")
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout)
            
            if thread.is_alive():
                return None  # Timeout
            
            return result[0] if result else None
        
        response_line = read_with_timeout(process, timeout=10)
        
        if response_line is None:
            print("❌ Response timeout (10 seconds)")
            
            # Check stderr for errors
            try:
                process.stderr.settimeout(1)
                stderr_data = process.stderr.read()
                if stderr_data:
                    print(f"STDERR: {stderr_data}")
            except:
                pass
            
            return False
        
        if not response_line.strip():
            print("❌ Empty response")
            return False
        
        print(f"✅ Raw response: {response_line.strip()}")
        
        # Try to parse JSON response
        try:
            response = json.loads(response_line.strip())
            print(f"✅ Parsed response: {json.dumps(response, indent=2)}")
            
            # Check if it's a valid MCP response
            if response.get("jsonrpc") == "2.0" and "id" in response:
                if "result" in response:
                    print("✅ Valid MCP success response")
                    return True
                elif "error" in response:
                    print(f"⚠️  MCP error response: {response['error']}")
                    return True  # Still valid MCP communication
                else:
                    print("❌ Invalid MCP response structure")
                    return False
            else:
                print("❌ Not a valid JSON-RPC 2.0 response")
                return False
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            print(f"Raw response: {response_line}")
            return False
            
    except Exception as e:
        print(f"❌ Communication error: {e}")
        return False
    
    finally:
        # Clean up
        print("\n🧹 Cleaning up...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        print("✅ Process terminated")


def test_direct_import():
    """Test direct import and basic functionality."""
    
    print("\n🔍 Direct Import Test")
    print("=" * 30)
    
    try:
        from src.core.mcp_server import server, run_mcp_server
        print("✅ MCP server import successful")
        
        # Test basic server info
        print(f"✅ Server name: {server.name}")
        print("✅ MCP server components loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment():
    """Test environment and dependencies."""
    
    print("\n🔍 Environment Test")
    print("=" * 25)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check required modules
    required_modules = [
        "asyncio", "json", "logging", "pathlib", "sys",
        "fastapi", "uvicorn", "pydantic", "openai", "mss", "PIL"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ Missing modules: {missing_modules}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file exists")
        
        # Check key configurations
        with open(env_file) as f:
            env_content = f.read()
            
        if "OPENAI_API_KEY=" in env_content:
            print("✅ OPENAI_API_KEY configured")
        else:
            print("⚠️  OPENAI_API_KEY not found in .env")
            
        if "API_KEY=" in env_content:
            print("✅ API_KEY configured")
        else:
            print("⚠️  API_KEY not found in .env")
    else:
        print("⚠️  .env file not found")
    
    return True


def main():
    """Run all debug tests."""
    
    print("🚀 ScreenMonitorMCP v2 - Debug Suite")
    print("=" * 50)
    
    # Test 1: Environment
    env_ok = test_environment()
    
    # Test 2: Direct import
    import_ok = test_direct_import()
    
    # Test 3: Stdio communication
    stdio_ok = test_mcp_stdio()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 20)
    print(f"Environment: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"Direct Import: {'✅ PASS' if import_ok else '❌ FAIL'}")
    print(f"Stdio Communication: {'✅ PASS' if stdio_ok else '❌ FAIL'}")
    
    if all([env_ok, import_ok, stdio_ok]):
        print("\n🎉 All tests passed! MCP server should work with clients.")
        print("\nTo use with MCP client:")
        print("1. Make sure 'disabled': false in your MCP config")
        print("2. Use absolute path for 'cwd' if needed")
        print("3. Check MCP client logs for detailed error messages")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        
        if not stdio_ok:
            print("\n🔧 Stdio Communication Troubleshooting:")
            print("- Check if mcp_main.py exists and is executable")
            print("- Verify Python path and working directory")
            print("- Look for import errors in stderr")
            print("- Try running 'python mcp_main.py' manually")


if __name__ == "__main__":
    main()