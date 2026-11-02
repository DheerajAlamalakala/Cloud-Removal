import argparse
import rasterio

def main():
    parser = argparse.ArgumentParser(description="Inspect one GeoTIFF's metadata and per-band range.")
    parser.add_argument("path", help="Path to a TIFF file")
    path = parser.parse_args().path

    with rasterio.open(path) as src:
        print("Width:", src.width)
        print("Height:", src.height)
        print("Bands:", src.count)
        print("Data types:", src.dtypes)
        print("CRS:", src.crs)
        print("Band descriptions:", src.descriptions)

        for i in range(1, src.count + 1):
            band = src.read(i)
            print(f"Band {i}: min={band.min()}, max={band.max()}")


if __name__ == "__main__":
    main()
