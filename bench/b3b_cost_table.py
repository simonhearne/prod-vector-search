"""B3b: extrapolate B3a's measured bytes/vector to bigger catalogues. Arithmetic, not a benchmark.
-> results/b3b.json"""
from vsearch.results import read_result, write_result


def cost_table(points: list[dict], n_products=(450_000, 10_000_000, 50_000_000),
               vectors=(1, 2), replicas=(1, 2, 3)) -> list[dict]:
    return [{"name": p["name"], "n_products": n, "vectors": v, "replicas": r,
             "gb": p["bytes_per_vector"] * n * v * r / 1e9}
            for p in points for n in n_products for v in vectors for r in replicas]


def main():
    b3a = read_result("b3a")
    write_result("b3b", {"rows": cost_table(b3a["payload"]["points"]),
                         "source": "b3a bytes_per_vector (Milvus internal_cache_loaded_bytes in RAM, location=memory, / rows; mmap excluded)",
                         "sample": b3a["meta"]["sample"]})


if __name__ == "__main__":
    main()
