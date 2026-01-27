import requests
import json
from datetime import datetime
import os

def test_session_stats_endpoint():
    """
    Test script to verify the /api/session/stats endpoint is working correctly.
    """
    base_url = "http://localhost:8000"
    endpoint = "/api/session/stats"
    
    print(f"Testing endpoint: {base_url}{endpoint}")
    
    # First, try to authenticate to get a valid token
    print("\n1. Attempting to login to get authentication token...")
    login_data = {
        "username": "admin",
        "password": "Admin@123"
    }
    
    try:
        login_response = requests.post(f"{base_url}/api/auth/login", json=login_data)
        print(f"Login status code: {login_response.status_code}")
        
        if login_response.status_code == 200:
            login_result = login_response.json()
            token = login_result.get('access_token')
            token_type = login_result.get('token_type', 'bearer')
            
            print("Login successful!")
            print(f"Token type: {token_type}")
            
            # Now test the session stats endpoint with the token
            print(f"\n2. Testing {endpoint} with authentication...")
            
            headers = {
                "Authorization": f"{token_type} {token}",
                "Content-Type": "application/json"
            }
            
            stats_response = requests.get(f"{base_url}{endpoint}", headers=headers)
            print(f"Session stats endpoint status code: {stats_response.status_code}")
            
            if stats_response.status_code == 200:
                print("SUCCESS: Session stats endpoint is working!")
                stats_data = stats_response.json()
                print(f"Response data: {json.dumps(stats_data, indent=2)}")
                return True
            else:
                print(f"FAILED: Session stats endpoint returned status {stats_response.status_code}")
                if stats_response.content:
                    print(f"Response: {stats_response.text}")
                return False
                
        else:
            print(f"FAILED: Login failed with status {login_response.status_code}")
            print(f"Response: {login_response.text}")
            print("\nTrying the endpoint without authentication to check if it's a routing issue...")
            
            # Test the endpoint without auth to see if it's a routing issue
            stats_response = requests.get(f"{base_url}{endpoint}")
            print(f"Session stats endpoint status code (no auth): {stats_response.status_code}")
            if stats_response.status_code != 404:
                print(f"Endpoint exists but requires authentication. Status: {stats_response.status_code}")
            else:
                print("Endpoint does not exist - likely a routing issue.")
                
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to {base_url}. Is the server running?")
        print("Please start the server with: uvicorn api:app --reload --port 8000")
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def check_routes():
    """
    Check all available routes in the FastAPI app
    """
    print("\n3. Checking available routes...")
    try:
        response = requests.get(f"http://localhost:8000/openapi.json")
        if response.status_code == 200:
            api_spec = response.json()
            paths = list(api_spec.get('paths', {}).keys())
            print("Available API routes:")
            for path in sorted(paths):
                methods = list(api_spec['paths'][path].keys())
                print(f"  {path} - {methods}")
            
            if '/api/session/stats' in paths:
                print("\n✓ /api/session/stats route exists in the API specification")
            else:
                print("\n✗ /api/session/stats route is missing from the API specification")
        else:
            print("Could not fetch OpenAPI spec")
    except Exception as e:
        print(f"Could not check routes: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING SESSION STATS ENDPOINT")
    print("=" * 60)
    
    # Check if server is running first
    try:
        health_check = requests.get("http://localhost:8000/health", timeout=5)
        if health_check.status_code == 200:
            print("✓ Server is running")
        else:
            print("✗ Server is not responding. Please start the server first.")
            print("Run: uvicorn api:app --reload --port 8000")
            exit(1)
    except requests.exceptions.ConnectionError:
        print("✗ Server is not running. Please start the server first.")
        print("Run: uvicorn api:app --reload --port 8000")
        exit(1)
    
    success = test_session_stats_endpoint()
    check_routes()
    
    print("\n" + "=" * 60)
    if success:
        print("RESULT: Endpoint is working correctly!")
    else:
        print("RESULT: Endpoint has issues that need to be addressed.")
    print("=" * 60)