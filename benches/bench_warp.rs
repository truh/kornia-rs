use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_rs::image::{Image, ImageSize};
use kornia_rs::resize::{meshgrid, InterpolationMode};
use kornia_rs::warp::{get_rotation_matrix2d, warp_affine};
use ndarray::stack;

use packed_simd::f32x4;

fn warp_native(
    image: &Image<f32, 3>,
    m: (f32, f32, f32, f32, f32, f32),
    new_size: ImageSize,
) -> Image<f32, 3> {
    // create the output image
    let mut output = Image::from_size_val(new_size, 0.0).unwrap();

    // create a grid of x and y coordinates for the output image
    // TODO: make this re-useable
    let x = ndarray::Array::range(0.0, new_size.width as f32, 1.0).insert_axis(ndarray::Axis(0));
    let y = ndarray::Array::range(0.0, new_size.height as f32, 1.0).insert_axis(ndarray::Axis(0));

    // create the meshgrid of x and y coordinates, arranged in a 2D grid of shape (height, width)
    let (xx, yy) = meshgrid(&x, &y);

    // TODO: benchmark this
    // stack the x and y coordinates into a single array of shape (height, width, 2)
    let xy = stack![ndarray::Axis(2), xx, yy];

    // let iterate over rows
    ndarray::Zip::from(xy.rows())
        .and(output.data.rows_mut())
        .for_each(|uv, mut out| {
            assert_eq!(uv.len(), 2);
            let (u, v) = (uv[0], uv[1]);

            // find corresponding position in src image
            let u_src = m.0 * u + m.1 * v + m.2;
            let v_src = m.3 * u + m.4 * v + m.5;

            if u_src < 0.0
                || v_src < 0.0
                || u_src >= image.size().width as f32 - 1.0
                || v_src >= image.size().height as f32 - 1.0
            {
                return;
            }

            let u0 = u_src.floor() as usize;
            let v0 = v_src.floor() as usize;
            let u1 = u0 + 1;
            let v1 = v0 + 1;

            //let w00 = (u1 as f32 - u_src) * (v1 as f32 - v_src);
            //let w01 = (u1 as f32 - u_src) * (v_src - v0 as f32);
            //let w10 = (u_src - u0 as f32) * (v1 as f32 - v_src);
            //let w11 = (u_src - u0 as f32) * (v_src - v0 as f32);
            let wa = f32x4::new(u1 as f32, u1 as f32, u_src, u_src);
            let wb = f32x4::new(v1 as f32, v_src, v1 as f32, v_src);
            let wc = f32x4::new(u_src, u_src, u0 as f32, u0 as f32);
            let wd = f32x4::new(v1 as f32, v_src, v1 as f32, v0 as f32);

            let w = (wa - wc) * (wb - wd);

            for c in (0..3).step_by(4) {
                let p00 = image.get_pixel(u0, v0, c).unwrap();
                let p01 = image.get_pixel(u0, v1, c).unwrap();
                let p10 = image.get_pixel(u1, v0, c).unwrap();
                let p11 = image.get_pixel(u1, v1, c).unwrap();

                let p = f32x4::new(p00, p01, p10, p11);

                let out = out.as_slice_mut().unwrap();
                out[c] = (p * w).sum();
            }
        });

    output
}

fn bench_warp_affine(c: &mut Criterion) {
    let mut group = c.benchmark_group("warp_affine");
    let image_sizes = vec![(256, 224), (512, 448), (1024, 896)];

    for (width, height) in image_sizes {
        let image_size = ImageSize { width, height };
        let id = format!("{}x{}", width, height);
        let image = Image::<u8, 3>::new(image_size, vec![0u8; width * height * 3]).unwrap();
        let image_f32 = image.clone().cast::<f32>().unwrap();
        let m = get_rotation_matrix2d((width as f32 / 2.0, height as f32 / 2.0), 45.0, 1.0);
        group.bench_with_input(BenchmarkId::new("par", &id), &image_f32, |b, i| {
            b.iter(|| warp_affine(black_box(i), m, image_size, InterpolationMode::Bilinear))
        });
        group.bench_with_input(BenchmarkId::new("par_rows", &id), &image_f32, |b, i| {
            b.iter(|| warp_native(black_box(i), m, image_size))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_warp_affine);
criterion_main!(benches);
