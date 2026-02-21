"""
GPU/CPU 자동 설정 스크립트 (최종 수정 버전)
- Protobuf 버전 충돌 문제 해결 (환경 변수 우선 설정)
- 중복 import 제거 및 로직 최적화
- GTX 1650Ti 등 4GB VRAM 기기 최적화
"""

import os

# [필독] TensorFlow를 불러오기 전에 반드시 실행해야 하는 설정
# 1. Protobuf 버전 충돌 우회 (TypeError 해결)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# 2. GPU 0번 사용 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 3. 불필요한 로그 억제 (0: 모두, 1: INFO 미출력, 2: WARNING 미출력, 3: ERROR만)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import platform
import tensorflow as tf
from tensorflow.keras import mixed_precision

print("=" * 70)
print("GPU/CPU 디바이스 설정 및 최적화 로드")
print("=" * 70)

# 1. 혼합 정밀도 설정 (4GB VRAM 필수 설정)
try:
    # 1650Ti 같은 4GB 모델은 메모리 부족이 잦으므로 float16 사용이 권장됩니다.
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("✅ 혼합 정밀도(Mixed Precision) 활성화 완료! (VRAM 절약 모드)")
except Exception as e:
    print(f"⚠️ 혼합 정밀도 설정 실패: {e}")

# 2. 시스템 정보 출력
print(f"\n[시스템 정보]")
print(f"Python: {sys.version.split()[0]}")
print(f"TensorFlow: {tf.__version__}")
print(f"운영체제: {platform.system()}")

# 3. GPU 설정 및 메모리 동적 할당
print(f"\n[GPU 설정]")
gpus = tf.config.list_physical_devices('GPU')
GPU_AVAILABLE = False
DEVICE_TYPE = "CPU"

if gpus:
    try:
        # GPU 메모리를 한꺼번에 점유하지 않고 필요한 만큼만 할당
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU 감지: {len(gpus)}개 (메모리 동적 할당 적용)")
        
        # 실제 연산 가능 여부 테스트
        with tf.device('/GPU:0'):
            test_a = tf.constant([[1.0, 2.0]])
            test_b = tf.constant([[3.0], [4.0]])
            _ = tf.matmul(test_a, test_b)
        
        print("✓ GPU 연산 테스트 성공")
        GPU_AVAILABLE = True
        DEVICE_TYPE = "GPU"
    except Exception as e:
        print(f"✗ GPU 감지되었으나 초기화 실패: {e}")
        print("→ CPU 모드로 전환합니다.")
else:
    # M1/M2 Mac 지원 확인
    if platform.system() == "Darwin" and "arm" in platform.processor().lower():
        print("✓ Apple Silicon 감지 - Metal(MPS) 가속 가능 여부를 확인하세요.")
        DEVICE_TYPE = "Metal/MPS"
    else:
        print("⚠ 사용 가능한 GPU가 없습니다. CPU 모드를 사용합니다.")

# 4. CuPy 설정 (선택사항)
print(f"\n[CuPy 가속 설정]")
CUPY_AVAILABLE = False
if GPU_AVAILABLE:
    try:
        import cupy as cp
        # 간단한 CuPy 연산 테스트
        cp.array([1, 2, 3])
        print(f"✓ CuPy 사용 가능 (버전: {cp.__version__})")
        CUPY_AVAILABLE = True
    except ImportError:
        print("⚠ CuPy 미설치 (필요 시 pip install cupy-cudaXXX)")
    except Exception as e:
        print(f"⚠ CuPy 로드 오류: {e}")

# 5. 최종 설정값 딕셔너리 생성
CONFIG = {
    'use_gpu': GPU_AVAILABLE,
    'device': DEVICE_TYPE,
    'mixed_precision': True,
    'cupy': CUPY_AVAILABLE
}

print("\n" + "=" * 70)
print(f"최종 확정 디바이스: {DEVICE_TYPE}")
print(f"메모리 최적화 상태: {'적용됨' if CONFIG['mixed_precision'] else '미적용'}")
print("=" * 70 + "\n")

# 다음 작업을 위해 CONFIG 객체를 반환하거나 사용할 수 있습니다.
