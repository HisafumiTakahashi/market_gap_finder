# 次のセッションでやること

## 即実行: e-Stat統合パイプライン

```bash
py scripts/integrate_estat.py --tag tokyo --top-n 20
```

### パイプライン処理フロー
1. `data/raw/tokyo_hotpepper.csv` 読み込み
2. ジャンル正規化 + JIS 3次メッシュコード付与
3. e-Stat APIから国勢調査メッシュ人口データ取得（1次メッシュ単位）
4. HotPepper競合データ × 人口データを3次メッシュで結合
5. scoring v2（人口ベース需要）で機会スコア算出
6. `data/processed/tokyo_integrated.csv` に保存 + 上位候補表示

### 2回目以降（API取得スキップ）
```bash
py scripts/integrate_estat.py --tag tokyo --top-n 20 --skip-fetch
```

---

## 今回のセッションで作成・変更したファイル

| ファイル | 状態 | 内容 |
|---------|------|------|
| `config/settings.py` | 変更 | ESTAT_API_KEY, ESTAT_API_BASE_URL 追加 |
| `src/collect/estat.py` | 新規 | e-Stat APIクライアント（統計表検索・データ取得・メッシュ人口） |
| `src/preprocess/mesh_converter.py` | 新規 | JIS標準3次メッシュ ↔ 緯度経度変換 |
| `src/analyze/scoring.py` | 変更 | v2スコアリング追加（人口ベース需要・重み付き機会スコア） |
| `scripts/integrate_estat.py` | 新規 | e-Stat統合パイプラインCLI |
| `notebooks/02_estat_demand_data.ipynb` | 変更 | settings.py/estat.pyモジュール利用に更新 |
| `notebooks/03_estat_eda.ipynb` | 新規 | e-Statデータ EDA（人口分布・空間分布・品質チェック） |
| `tests/test_estat.py` | 新規 | estat.py基本ユニットテスト（6 passed） |

---

## その後のステップ（優先順）

### P1: 統合結果の検証
- `tokyo_integrated.csv` の中身をEDAで確認
- v1スコアとv2スコアの比較分析
- 人口データの結合率・欠損確認

### P2: ダッシュボード更新
- `src/visualize/dashboard.py` を v2スコア対応に更新
- 人口ヒートマップの追加
- v1/v2切り替えUIの追加

### P3: モデル高度化
- 空間特徴量の追加（最寄り駅距離、商業施設密度）
- 時系列要素（人口増減トレンド）
- LightGBM等の機械学習モデル検討

### P4: バックテスト
- 既存店の成功/撤退データとスコアの相関分析
- 時系列ホールドアウトでの予測精度評価

---

## 環境変数（設定済み）
- `HOTPEPPER_API_KEY`: settings.pyにデフォルト値設定済み
- `ESTAT_API_KEY`: settings.pyにデフォルト値設定済み
