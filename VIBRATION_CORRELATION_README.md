# Vibration Correlation Plot Tool

진동(vibration) 특성과 센서 데이터 간의 상관관계를 시각화하는 도구입니다.

## 기능

1. **상관관계 히트맵 (Correlation Heatmap)**: 모든 특성 간의 피어슨/스피어만 상관계수를 히트맵으로 시각화
2. **산점도 행렬 (Scatter Matrix)**: 진동 특성 간의 관계를 산점도와 밀도 플롯으로 표현
3. **Top 상관관계 (Top Correlations)**: 진동-센서 간 가장 높은 상관관계를 가진 쌍을 산점도로 표시
4. **RUL vs 진동 (RUL Correlation)**: 잔여 수명(RUL)과 진동 특성 간의 관계 분석

## 설치

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 사용법

### 1. 기본 사용 (진동 특성만)

```bash
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1
```

생성되는 플롯:
- 진동 특성 간 상관관계 히트맵
- 진동 특성 간 산점도 행렬
- RUL vs 진동 특성 플롯

### 2. 센서 포함 분석

```bash
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --include-sensors
```

추가 생성:
- 진동 + 센서 특성 전체 히트맵 (25개 특성)
- 진동-센서 간 Top N 상관관계 산점도

### 3. 특정 플롯만 생성

```bash
# 히트맵만
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --plot-type heatmap

# 산점도 행렬만
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --plot-type scatter

# Top 상관관계만 (센서 포함 필요)
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --include-sensors --plot-type top-corr

# RUL 분석만
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --plot-type rul
```

### 4. 고급 옵션

```bash
# 스피어만 상관계수 사용
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --corr-method spearman

# Top 12개 상관관계 표시
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --include-sensors --n-top 12

# 샘플링 크기 조정 (성능 향상)
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --sample-size 10000

# 커스텀 출력 디렉토리
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --output-dir ./my_plots

# 테스트 데이터 분석
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --split test
```

## 출력 파일

기본적으로 `./correlation_plots/` 디렉토리에 다음 파일들이 생성됩니다:

1. **`{tag}_corr_heatmap.png`**: 상관관계 히트맵
   - 삼각형 마스크 적용 (중복 제거)
   - 상관계수 값 표시
   - -1 ~ +1 컬러맵 (coolwarm)

2. **`{tag}_scatter_matrix.png`**: 진동 특성 산점도 행렬
   - 대각선: KDE 밀도 플롯
   - 비대각선: 산점도 (5000개 샘플)

3. **`{tag}_top_vib_sensor_corr.png`**: Top N 진동-센서 상관관계
   - 각 서브플롯에 추세선 포함
   - 상관계수 값 표시

4. **`{tag}_rul_vs_vibration.png`**: RUL과 진동 특성 간 관계
   - 4개 진동 특성 각각에 대한 산점도
   - 상관계수 값 표시

5. **`{tag}_corr_report.csv`**: 모든 특성 쌍의 상관계수 (CSV)
   - 절대값 기준 내림차순 정렬
   - Feature_1, Feature_2, Correlation, Abs_Correlation 컬럼

여기서 `{tag}`는 `{set_id}_{version}_run{run}_{split}` 형식입니다.

## 주요 발견 (예시)

위 명령어 실행 결과, 다음과 같은 Top 진동-센서 상관관계를 발견:

```
fe_RMS <-> sensor16 : 0.826
fe_RMS <-> sensor5  : 0.821
fe_RMS <-> sensor6  : 0.808
fe_RMS <-> sensor1  : 0.795
fe_RMS <-> sensor21 : 0.792
fe_RMS <-> sensor20 : 0.792
fe_RMS <-> sensor7  : 0.785
fe_RMS <-> sensor12 : 0.785
fe_RMS <-> sensor10 : 0.697
```

→ **fe_RMS**(진동 RMS 값)가 다수의 센서와 강한 양의 상관관계를 보임

## CLI 옵션 전체 목록

```
Data Selection:
  --set-id          Dataset ID (default: 200)
  --version         Dataset version (default: v1)
  --run             Run number (default: 1)
  --split           Data split: train or test (default: train)

Plot Options:
  --plot-type       Plot type: heatmap, scatter, top-corr, rul, all (default: all)
  --include-sensors Include sensor features in analysis
  --n-top           Number of top correlations to plot (default: 6)
  --sample-size     Sample size for scatter plots (default: 5000)
  --corr-method     Correlation method: pearson, spearman, kendall (default: pearson)

Output Options:
  --output-dir      Directory to save plots (default: ./correlation_plots)
  --show            Display plots in addition to saving
```

## 예시 워크플로우

### 단계 1: 빠른 탐색 (진동 특성만)
```bash
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --plot-type heatmap
```

### 단계 2: 센서와의 관계 확인
```bash
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --include-sensors --plot-type top-corr --n-top 12
```

### 단계 3: RUL 관계 분석
```bash
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --plot-type rul
```

### 단계 4: 전체 리포트 생성
```bash
python plot_vibration_correlation.py --set-id 200 --version v1 --run 1 --include-sensors
```

## 참고사항

- **샘플링**: 산점도 생성 시 성능을 위해 기본 5000개 샘플 사용 (조정 가능)
- **상관계수**: 기본 피어슨 상관계수, 비선형 관계는 스피어만 추천
- **메모리**: `--include-sensors` 사용 시 메모리 사용량 증가 (25x25 상관 행렬)
- **백엔드**: Matplotlib Agg 백엔드 사용 (서버 환경에서도 동작)

## 트러블슈팅

### ModuleNotFoundError
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 파일을 찾을 수 없음
- `data/training/`, `data/test/` 구조 확인
- `--set-id`, `--version`, `--run` 값이 실제 파일명과 일치하는지 확인

### 메모리 부족
```bash
# 샘플 크기 줄이기
python plot_vibration_correlation.py ... --sample-size 1000
```

## 라이선스

MIT
