"""B3c: the re-embed bill. Hours and $ to re-embed the catalogue at measured throughput.
-> results/b3c.json"""
import argparse

from vsearch.results import read_result, write_result


def reembed_bill(items_per_s: float, n: int, usd_per_hour: float) -> dict:
    hours = n / items_per_s / 3600
    return {"hours": hours, "usd": hours * usd_per_hour}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-usd-per-hour", type=float, default=0.0)
    ap.add_argument("--colab-items-per-s", type=float, default=None)
    a = ap.parse_args(argv)
    t = read_result("b3c_throughput")["payload"]
    rows = []
    for label, rate, usd in [(f"local {t.get('device')} text", t.get("text_items_per_s"), a.gpu_usd_per_hour),
                             (f"local {t.get('device')} image", t.get("image_items_per_s"), a.gpu_usd_per_hour),
                             ("colab gpu text", a.colab_items_per_s, a.gpu_usd_per_hour)]:
        if rate:
            for n in (450_000, 10_000_000):
                rows.append({"setup": label, "items_per_s": rate, "n": n, **reembed_bill(rate, n, usd)})
    write_result("b3c", {"rows": rows, "usd_per_hour": a.gpu_usd_per_hour,
                         "note": "usd_per_hour=0 means local hardware (cost not priced)"})


if __name__ == "__main__":
    main()
