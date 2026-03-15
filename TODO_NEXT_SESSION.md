# 次のセッションでやること

## 現在のステータス

- **ブランチ**: `feature/add-docstrings` (pushed)
- **パイプライン**: e-Stat 2020年国勢調査メッシュ人口統合済み（東京）
- **スコアリング**: v1(店舗数) / v2(人口) / v3(空間特徴量) 実装済み
- **ダッシュボード**: v1/v2切替 + 人口ヒートマップ対応済み
- **テスト**: 51件全パス

### 即実行コマンド（キャッシュ利用）
```bash
PYTHONPATH=. py scripts/integrate_estat.py --tag tokyo --top-n 20 --skip-fetch
PYTHONPATH=. py scripts/validate_scores.py --tag tokyo --top-n 20
PYTHONPATH=. py -c "from src.visualize.dashboard import run_dashboard; run_dashboard(tag='tokyo')"
```

---

## P1: ダッシュボードにv3追加（即対応）

- `dashboard.py` の Score Version ドロップダウンに `v3 (空間特徴量)` を追加
- v3選択時は `add_all_features` → `compute_opportunity_score_v3` で再計算
- 空間特徴量（genre_diversity, saturation_index等）のツールチップ表示

---

## P2: 他エリア展開（大阪・名古屋）

### 目的
東京以外のエリアでパイプラインの汎化性を確認する。

### 手順
1. `scripts/collect_area.py` で大阪・名古屋のHotPepperデータ収集
2. `scripts/integrate_estat.py --tag osaka` で統合パイプライン実行
   - 大阪の1次メッシュ（5135等）に対応する statsDataId が動的検索で取得できるか確認
3. `scripts/validate_scores.py --tag osaka` でスコア検証
4. エリア間のスコア分布・特徴量分布を比較

---

## P3: 外部データ追加（モデルの質向上）

### 駅乗降客数
- 国土数値情報の鉄道データ（GeoJSON）をダウンロード
- 各メッシュから最寄り駅への距離を計算 → `nearest_station_distance` 特徴量
- 駅の乗降客数 → `station_passengers` 特徴量

### 地価公示
- 国土交通省 地価公示データ（CSV）
- メッシュ単位で平均地価を結合 → `land_price` 特徴量
- `scoring.py` の `WEIGHT_LAND_PRICE` を有効化

---

## P4: 教師あり学習モデル

### 目的
ルールベースのスコアから、データ駆動のML予測モデルへ移行する。

### ラベル設計（開店成功度プロキシ）
- HotPepperの `avg_rating` × `total_reviews` を成功度スコアとして利用
- 高評価 & 高レビュー数 = そのジャンル × エリアが成功しやすい

### 特徴量
- 人口、店舗数、ジャンル多様性、HHI、隣接競合、飽和度
- （P3完了後）最寄り駅距離、地価

### モデル
- LightGBM で回帰 or ランキング学習
- 5-fold CV + SHAP で特徴量重要度を可視化
- v3ルールベースとの比較

---

## P5: レポート自動生成

- 上位候補を地図付きHTMLレポートとして出力
- メッシュ×ジャンルごとに「なぜここが有望か」を特徴量ベースで自動説明
- FC本部・営業チーム向けの成果物として利用可能に

---

## 既知の課題

| 課題 | 詳細 |
|------|------|
| `neighbor_avg_restaurants` 相関N/A | validate_scores.py で当該列の正規化時に分散ゼロ。NaN伝播を修正する |
| v1テストの `base_df` に `mesh_code` 必須 | `compute_demand_score` が `mesh_code` を groupby キーに使うため |
| `backtest.py` が未使用 | 過去の開店履歴データがないと動作しない。P4でラベル設計後に活用 |

---

## 環境変数（設定済み）
- `HOTPEPPER_API_KEY`: settings.py にデフォルト値設定済み
- `ESTAT_API_KEY`: settings.py にデフォルト値設定済み

## プロジェクト構造（更新後）

```
config/settings.py          # API設定・パラメータ一元管理
src/collect/
  hotpepper.py              # HotPepper APIクライアント
  estat.py                  # e-Stat APIクライアント（動的statsDataId検索）
  google_places.py          # Google Places（未使用）
src/preprocess/
  cleaner.py                # データクレンジング
  mesh_converter.py         # JIS 3次メッシュ変換
src/analyze/
  features.py               # 空間特徴量（多様性・HHI・隣接競合・飽和度）
  scoring.py                # v1/v2/v3 スコアリング
  backtest.py               # バックテスト（未使用）
src/visualize/
  dashboard.py              # Dash ダッシュボード
scripts/
  collect_area.py           # データ収集CLI
  integrate_estat.py        # e-Stat統合パイプライン（v3対応）
  validate_scores.py        # スコア検証・比較ツール
tests/
  test_estat.py             # 6 tests
  test_features.py          # 15 tests
  test_scoring.py           # 30 tests
notebooks/
  01_collection_eda.ipynb   # HotPepper EDA
  02_estat_demand_data.ipynb # e-Stat需要データ
  03_estat_eda.ipynb        # e-Stat EDA
```
