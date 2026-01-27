#!/usr/bin/env python
"""
Comprehensive API Testing Script for EcoPredict
Tests all endpoints and generates a detailed report
"""

import requests
import json
import sys
from datetime import datetime
from typing import Dict, List, Any

# API endpoint
BASE_URL = "http://localhost:8000"
TIMEOUT = 5  # seconds

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class APITester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
        self.session = requests.Session()
        
    def print_header(self, text: str):
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}{text}{RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")
    
    def print_pass(self, text: str):
        print(f"{GREEN}✓ PASS:{RESET} {text}")
    
    def print_fail(self, text: str):
        print(f"{RED}✗ FAIL:{RESET} {text}")
    
    def print_warning(self, text: str):
        print(f"{YELLOW}⚠ WARNING:{RESET} {text}")
    
    def test_health_endpoint(self):
        """Test /health endpoint"""
        self.print_header("Testing Health Endpoints")
        
        # Basic health check
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                self.print_pass(f"Health check returned status: {data.get('status')}")
                self.results.append({
                    "endpoint": "/health",
                    "status": "PASS",
                    "code": 200
                })
            else:
                self.print_fail(f"Health check returned {response.status_code}")
                self.results.append({
                    "endpoint": "/health",
                    "status": "FAIL",
                    "code": response.status_code
                })
        except Exception as e:
            self.print_fail(f"Health check failed: {e}")
            self.results.append({
                "endpoint": "/health",
                "status": "FAIL",
                "error": str(e)
            })
        
        # Detailed health check
        try:
            response = self.session.get(
                f"{self.base_url}/health/detailed",
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                self.print_pass("Detailed health check successful")
                self.results.append({
                    "endpoint": "/health/detailed",
                    "status": "PASS",
                    "code": 200
                })
            else:
                self.print_fail(f"Detailed health check returned {response.status_code}")
                self.results.append({
                    "endpoint": "/health/detailed",
                    "status": "FAIL",
                    "code": response.status_code
                })
        except Exception as e:
            self.print_fail(f"Detailed health check failed: {e}")
            self.results.append({
                "endpoint": "/health/detailed",
                "status": "FAIL",
                "error": str(e)
            })
        
        # Readiness check
        try:
            response = self.session.get(
                f"{self.base_url}/ready",
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                self.print_pass("Readiness check passed")
                self.results.append({
                    "endpoint": "/ready",
                    "status": "PASS",
                    "code": 200
                })
            else:
                self.print_warning(f"Readiness check returned {response.status_code}")
                self.results.append({
                    "endpoint": "/ready",
                    "status": "WARNING",
                    "code": response.status_code
                })
        except Exception as e:
            self.print_fail(f"Readiness check failed: {e}")
            self.results.append({
                "endpoint": "/ready",
                "status": "FAIL",
                "error": str(e)
            })
    
    def test_metrics_endpoint(self):
        """Test /metrics endpoint"""
        self.print_header("Testing Metrics Endpoint")
        
        try:
            response = self.session.get(
                f"{self.base_url}/metrics",
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                memory = data.get('memory', {})
                cpu = data.get('cpu', {})
                
                self.print_pass("Metrics endpoint working")
                self.print_pass(f"  Memory: {memory.get('percent', 'N/A'):.1f}% used")
                self.print_pass(f"  CPU: {cpu.get('percent', 'N/A'):.1f}%")
                self.print_pass(f"  Uptime: {data.get('uptime_seconds', 0):.0f}s")
                
                self.results.append({
                    "endpoint": "/metrics",
                    "status": "PASS",
                    "code": 200
                })
            else:
                self.print_fail(f"Metrics endpoint returned {response.status_code}")
                self.results.append({
                    "endpoint": "/metrics",
                    "status": "FAIL",
                    "code": response.status_code
                })
        except Exception as e:
            self.print_fail(f"Metrics endpoint failed: {e}")
            self.results.append({
                "endpoint": "/metrics",
                "status": "FAIL",
                "error": str(e)
            })
    
    def test_connectivity(self):
        """Test basic connectivity"""
        self.print_header("Testing Basic Connectivity")
        
        try:
            response = self.session.get(
                f"{self.base_url}/",
                timeout=TIMEOUT
            )
            
            if response.status_code in [200, 404]:
                self.print_pass(f"Server is reachable (HTTP {response.status_code})")
                self.results.append({
                    "test": "connectivity",
                    "status": "PASS",
                    "code": response.status_code
                })
            else:
                self.print_warning(f"Server returned {response.status_code}")
                self.results.append({
                    "test": "connectivity",
                    "status": "WARNING",
                    "code": response.status_code
                })
        except requests.ConnectionError:
            self.print_fail(f"Cannot connect to {self.base_url}")
            self.results.append({
                "test": "connectivity",
                "status": "FAIL",
                "error": "Connection refused"
            })
        except Exception as e:
            self.print_fail(f"Connectivity test failed: {e}")
            self.results.append({
                "test": "connectivity",
                "status": "FAIL",
                "error": str(e)
            })
    
    def print_summary(self):
        """Print test summary"""
        self.print_header("Test Summary")
        
        passed = sum(1 for r in self.results if r.get('status') == 'PASS')
        failed = sum(1 for r in self.results if r.get('status') == 'FAIL')
        warned = sum(1 for r in self.results if r.get('status') == 'WARNING')
        total = len(self.results)
        
        print(f"Total Tests: {total}")
        print(f"{GREEN}Passed: {passed}{RESET}")
        print(f"{RED}Failed: {failed}{RESET}")
        print(f"{YELLOW}Warnings: {warned}{RESET}")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        # Generate JSON report
        report = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "warnings": warned,
                "success_rate": f"{success_rate:.1f}%"
            },
            "results": self.results
        }
        
        # Save report
        report_file = "api_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {report_file}")
        
        return failed == 0
    
    def run_all_tests(self):
        """Run all tests"""
        print(f"{BLUE}EcoPredict API Test Suite{RESET}")
        print(f"Testing: {self.base_url}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self.test_connectivity()
        self.test_health_endpoint()
        self.test_metrics_endpoint()
        
        success = self.print_summary()
        
        return success


def main():
    """Main entry point"""
    try:
        tester = APITester()
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Test interrupted by user{RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{RED}Test suite failed: {e}{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()