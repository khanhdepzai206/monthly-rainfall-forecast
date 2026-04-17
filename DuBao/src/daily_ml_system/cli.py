# -*- coding: utf-8 -*-
"""
CLI: train | predict | log | actual | evaluate | retrain

Chạy từ thư mục DuBao/src:
  python -m daily_ml_system.cli train
  python -m daily_ml_system.cli predict
  python -m daily_ml_system.cli log
  python -m daily_ml_system.cli actual --date 2024-07-15 --value 12.3
  python -m daily_ml_system.cli evaluate
  python -m daily_ml_system.cli retrain
"""
import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description="Hệ thống dự đoán mưa ngày mai (RF, XGBoost, LR)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="Huấn luyện 3 mô hình và lưu triple_models.pkl")

    sub.add_parser("predict", help="In dự đoán ngày mai (không ghi log)")

    sp = sub.add_parser("log", help="Dự đoán ngày mai và ghi prediction_log.csv")
    sp.add_argument("--json", action="store_true", help="In JSON")

    sp = sub.add_parser("actual", help="Nhập mưa thực tế cho một target_date đã dự đoán")
    sp.add_argument("--date", required=True, help="YYYY-MM-DD (ngày cần nhập mưa thực tế)")
    sp.add_argument("--value", type=float, required=True, help="Lượng mưa thực tế (mm)")

    sub.add_parser("evaluate", help="MAE 7 ngày + mô hình tốt nhất")

    sp = sub.add_parser("retrain", help="Kiểm tra ngưỡng MAE và retrain nếu cần")
    sp.add_argument("--threshold", type=float, default=None, help="Ngưỡng MAE (mặc định từ config)")

    args = p.parse_args(argv)

    if args.cmd == "train":
        from .train import train_all
        train_all()
        return 0

    if args.cmd == "predict":
        from .predict_daily import predict_tomorrow
        t, p, preds, probs = predict_tomorrow()
        print(f"Ngày dự đoán mưa (target): {t.date()}")
        print(f"Feature đến ngày: {p.date()}")
        print(f"RandomForest: {preds['rf']:.2f} mm (p={probs['rf']:.2f})")
        print(f"XGBoost:      {preds['xgb']:.2f} mm (p={probs['xgb']:.2f})")
        print(f"ExtraTrees:   {preds['et']:.2f} mm (p={probs['et']:.2f})")
        return 0

    if args.cmd == "log":
        from .log_io import run_daily_predict_and_log
        out = run_daily_predict_and_log()
        if getattr(args, "json", False):
            print(json.dumps(out, indent=2, default=str))
        else:
            print("Đã ghi log:", out["log"])
            print("target_date:", out["target_date"], "preds:", out["preds"])
        return 0

    if args.cmd == "actual":
        from .log_io import update_actual
        row = update_actual(args.date, args.value)
        print("Đã cập nhật:")
        print(row.to_string())
        return 0

    if args.cmd == "evaluate":
        from .evaluate import full_report
        r = full_report()
        print(json.dumps(r, indent=2, default=str))
        return 0

    if args.cmd == "retrain":
        from .retrain import check_and_retrain
        check_and_retrain(threshold=args.threshold)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
