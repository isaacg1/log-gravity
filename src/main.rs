use image::{ImageError, Rgb, RgbImage};
use rand::prelude::*;
use rand_distr::LogNormal;

type Color = [f64; 3];

type Location = [f64; 2];

struct Object {
    center: Location,
    mass: f64,
    color: Color,
    velocity: [f64; 2],
}
impl Object {
    fn sample<R: Rng>(rng: &mut R, mass_distr: LogNormal<f64>, size: u32) -> Object {
        let color = [
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ];
        let center = [
            rng.gen_range(0.0..size as f64),
            rng.gen_range(0.0..size as f64),
        ];
        let mass = mass_distr.sample(rng);
        Object {
            color,
            center,
            mass,
            velocity: [0.0; 2],
        }
    }
}

const TIME_STEP: f64 = 0.001;
const GRAV_CONST: f64 = 10.0;
const MAX_FORCE: f64 = 10.0;
const FRICTION: f64 = 0.1;
fn make_image(
    num_objects: usize,
    size: u32,
    log_mean_mass: f64,
    log_sdev_mass: f64,
    updates_per_object: usize,
    num_steps: usize,
    sigmoid_param: f64,
    seed: u64,
) -> RgbImage {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut det_img: Vec<Vec<Color>> = vec![vec![[0.0; 3]; size as usize]; size as usize];
    let mass_distr = LogNormal::new(log_mean_mass, log_sdev_mass).unwrap();
    let mut objects: Vec<Object> = (0..num_objects)
        .map(|_| Object::sample(&mut rng, mass_distr, size))
        .collect();
    let num_updates_per_step = num_objects * updates_per_object;
    for _ in 0..num_steps {
        for _ in 0..num_updates_per_step {
            // Update velocity
            let object1_index = rng.gen_range(0..num_objects);
            let object2_index = rng.gen_range(0..num_objects);
            if object1_index == object2_index {
                continue;
            }
            let object1 = &objects[object1_index];
            let object2 = &objects[object2_index];
            let offset: Vec<f64> = object1
                .center
                .iter()
                .zip(&object2.center)
                .map(|(p1, p2)| {
                    let off1 = p1 - p2;
                    if off1 < -(size as f64) / 2.0 {
                        off1 + size as f64
                    } else if off1 > size as f64 / 2.0 {
                        off1 - size as f64
                    } else {
                        off1
                    }
                })
                .collect();
            let radius_sq: f64 = offset.iter().map(|p| p.powi(2)).sum();
            let radius = radius_sq.sqrt();
            let color_sim: f64 = object1
                .color
                .iter()
                .zip(&object2.color)
                .map(|(&c1, &c2)| c1 * c2)
                .sum();
            let force =
                (GRAV_CONST * object1.mass * object2.mass * color_sim / radius_sq).min(MAX_FORCE);
            let accel_mag1 = force / object1.mass;
            let accel_mag2 = force / object2.mass;
            let unit1: Vec<f64> = offset.iter().map(|p| -p / radius).collect();
            let unit2: Vec<f64> = offset.iter().map(|p| p / radius).collect();
            let dv1: Vec<f64> = unit1.iter().map(|p| p * accel_mag1).collect();
            let dv2: Vec<f64> = unit2.iter().map(|p| p * accel_mag2).collect();
            let object1_mut = &mut objects[object1_index];
            object1_mut.velocity[0] += dv1[0];
            object1_mut.velocity[1] += dv1[1];
            let object2_mut = &mut objects[object2_index];
            object2_mut.velocity[0] += dv2[0];
            object2_mut.velocity[1] += dv2[1];
        }
        for object in &mut objects {
            // Update position
            object.center[0] += TIME_STEP * object.velocity[0];
            object.center[1] += TIME_STEP * object.velocity[1];
            if object.center[0] >= size as f64 {
                object.center[1] -= size as f64;
            }
            if object.center[0] < 0.0 {
                object.center[0] += size as f64;
            }
            if object.center[1] >= size as f64 {
                object.center[1] -= size as f64;
            }
            if object.center[1] < 0.0 {
                object.center[1] += size as f64;
            }
            // Friction
            let speed: f64 = object
                .velocity
                .iter()
                .map(|v| v.powi(2))
                .sum::<f64>()
                .sqrt();
            if speed < FRICTION {
                object.velocity = [0.0; 2];
            } else {
                let unit_velocity = object.velocity.map(|v| v / speed);
                object.velocity[0] -= unit_velocity[0] * FRICTION;
                object.velocity[1] -= unit_velocity[1] * FRICTION;
            }
            // Draw
            let radius = (object.mass.sqrt() * 5.0).min(2.0);
            let iradius = radius.powi(2) as i32;
            let center = object.center.map(|p| p as i32);
            for dr in -iradius..=iradius {
                for dc in -iradius..=iradius {
                    if dr.pow(2) + dc.pow(2) > iradius {
                        continue;
                    }
                    let row = (center[0] + dr + size as i32) as u32 % size;
                    let col = (center[1] + dc + size as i32) as u32 % size;
                    det_img[row as usize][col as usize]
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, c)| *c += object.color[i]);
                }
            }
        }
    }
    let mut img = RgbImage::from_pixel(size, size, Rgb([128; 3]));
    for r in 0..size {
        for c in 0..size {
            let det_color = det_img[r as usize][c as usize]
                .map(|channel| (255.0 / (1.0 + (channel / sigmoid_param).exp())) as u8);
            img.put_pixel(r, c, Rgb(det_color));
        }
    }
    img
}

fn main() -> Result<(), ImageError> {
    let num_objects = 400;
    let size = 2000;
    let log_mean_mass = 3;
    let log_sdev_mass = 4;
    let updates_per_object = 5;
    let num_steps = 100000;
    let sigmoid_param = 10;
    let seed = 0;
    let filename = format!(
        "img-{}-{}-{}-{}-{}-{}-{}-{}.png",
        num_objects, size, log_mean_mass, log_sdev_mass, updates_per_object, num_steps, sigmoid_param, seed
    );
    println!("{}", filename);
    let img = make_image(
        num_objects,
        size,
        log_mean_mass as f64,
        log_sdev_mass as f64,
        updates_per_object,
        num_steps,
        sigmoid_param as f64,
        seed,
    );
    img.save(filename)
}
