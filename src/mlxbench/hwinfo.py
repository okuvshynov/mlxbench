# Copyright © 2025 Oleksandr Kuvshynov

import platform
import subprocess
import re
from typing import Dict


def apple_hwinfo() -> Dict:
    """
    Comprehensive Apple Silicon detection and specification gathering
    """
    result = {
        'platform_processor': platform.processor(),
        'platform_machine': platform.machine(),
        'is_apple_silicon': False,
        'chip_details': None,
        'performance_cores': None,
        'efficiency_cores': None,
        'gpu_core_count': None,
        'total_memory_gb': None
    }
    
    # Check for Apple Silicon using multiple methods
    if platform.processor() == 'arm':
        result['is_apple_silicon'] = True
    
    # Get detailed specs via sysctl
    try:
        brand_result = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, check=True
        )
        brand = brand_result.stdout.strip()
        
        if 'Apple' in brand:
            result['is_apple_silicon'] = True
            result['chip_details'] = brand
            
            # Get core counts
            try:
                perf_result = subprocess.run(
                    ['sysctl', '-n', 'hw.perflevel0.logicalcpu_max'],
                    capture_output=True, text=True, check=True
                )
                result['performance_cores'] = int(perf_result.stdout.strip())
                
                eff_result = subprocess.run(
                    ['sysctl', '-n', 'hw.perflevel1.logicalcpu_max'],
                    capture_output=True, text=True, check=True
                )
                result['efficiency_cores'] = int(eff_result.stdout.strip())
            except:
                pass
            
            # Get memory
            try:
                mem_result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True, text=True, check=True
                )
                memory_bytes = int(mem_result.stdout.strip())
                result['total_memory_gb'] = round(memory_bytes / (1024**3), 1)
            except:
                pass
            
            # Get GPU core count
            try:
                gpu_result = subprocess.run(
                    'ioreg -l | grep gpu-core-count',
                    shell=True,
                    capture_output=True, text=True, check=True
                )
                # Parse line like: | |   |   |   "gpu-core-count" = 76
                match = re.search(r'"gpu-core-count"\s*=\s*(\d+)', gpu_result.stdout)
                if match:
                    result['gpu_core_count'] = int(match.group(1))
            except:
                pass
    except:
        pass
    
    return result


def format_hwinfo(hwinfo: Dict) -> str:
    """
    Format hardware info into a concise string showing model, memory, and GPU cores
    """
    if not hwinfo.get('is_apple_silicon'):
        return "Non-Apple Silicon Mac"
    
    # Extract model name from chip details
    chip = hwinfo.get('chip_details', 'Unknown')
    if 'Apple' in chip:
        # Extract just the model part (e.g., "M2 Ultra" from "Apple M2 Ultra")
        model = chip.replace('Apple ', '')
    else:
        model = chip
    
    memory = hwinfo.get('total_memory_gb', 0)
    gpu_cores = hwinfo.get('gpu_core_count', 0)
    
    return f"{model} | {memory}GB | {gpu_cores} GPU cores"


if __name__ == '__main__':
    info = apple_hwinfo()
    print(info)
    print(f"\nFormatted: {format_hwinfo(info)}")