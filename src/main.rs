use std::fs::File;
use std::io::prelude::*;
use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use std::ops::Div;
use std::f32::consts::PI;

#[derive(Debug, Clone)]
struct Vector {
    x: f32,
    y: f32,
    z: f32,
}

struct Ray {
    orig: Vector,
    dest: Vector,
}

struct Sphere {
    orig: Vector,
    radius: f32,
    albedo: Vector,
}

struct Scene {
    spheres: Vec<Sphere>,
}

impl Add for Vector {
    type Output = Vector;

    fn add(self, other: Vector) -> Vector {
        Vector { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
}

impl Sub for Vector {
    type Output = Vector;

    fn sub(self, other: Vector) -> Vector {
        Vector { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
}

impl Mul<f32> for Vector {
    type Output = Vector;

    fn mul(self, scalar: f32) -> Vector {
        Vector { x: scalar*self.x, y: scalar*self.y, z: scalar*self.z}
    }
}

impl Mul<Vector> for Vector {
    type Output = Vector;

    fn mul(self, other: Vector) -> Vector {
        Vector { x: other.x*self.x, y: other.y*self.y, z: other.z*self.z}
    }
}

impl Div<f32> for Vector {
    type Output = Vector;

    fn div(self, scalar: f32) -> Vector {
        Vector { x: self.x/scalar, y: self.y/scalar, z: self.z/scalar}
    }
}

impl Vector {
    fn dot(&self, other: &Vector) -> f32 {
        self.x*other.x + self.y*other.y + self.z*other.z
    }

    fn getNorm2(&self) -> f32 {
        self.dot(&self)
    }

    fn normalize(&mut self) {
        let norm = (self.getNorm2() as f64).sqrt() as f32;
        self.x /= norm;
        self.y /= norm;
        self.z /= norm;
    }

    fn getNormalized(&mut self) -> Vector {
        let mut clone: Vector = self.clone();
        clone.normalize();
        clone
    }
}

impl Sphere {

    fn intersection(&self, r: &Ray, P: &mut Vector, N: &mut Vector, t: &mut f32) -> bool {
        // résoud a*t^2 + b*t + c = 0

        let diff: Vector = r.orig.clone() - self.orig.clone();
        let a: f32 = 1.0;
        let b: f32 = 2.0 * r.dest.clone().dot(&diff);
        let c: f32 = diff.getNorm2() - self.radius*self.radius;

        let delta: f32 = b*b - 4.0*a*c;
        if delta < 0f32 {
            false
        } else {
            let t1: f32 = (-b - delta.sqrt())/(2f32*a);
            let t2: f32 = (-b + delta.sqrt())/(2f32*a);
            if t2 < 0f32 {
                false
            } else {
                if t1 > 0.0 {
                    *t = t1;
                } else {
                    *t = t2;
                }

                let p = r.orig.clone() + r.dest.clone()* *t;
                P.x = p.x;
                P.y = p.y;
                P.z = p.z;
                let normale = (p - self.orig.clone()).getNormalized();
                N.x = normale.x;
                N.y = normale.y;
                N.z = normale.z;
                true
            }
        }

    }

}

impl Scene {

    fn addSphere(&mut self, sphere: Sphere) {
        self.spheres.push(sphere);
    }

    fn intersection(&self, r: &Ray, P: &mut Vector, N: &mut Vector, sphere_id: &mut usize, min_t: &mut f32) -> bool {
        let mut has_inter: bool = false;
        *min_t = 1e99;

        for (i, s) in self.spheres.iter().enumerate() {
            let mut t: f32 = 0.0;
            let mut local_p: Vector = Vector {x: 0.0, y: 0.0, z: 0.0};
            let mut local_n: Vector = Vector {x: 0.0, y: 0.0, z: 0.0};
            let local_has_inter: bool = s.intersection(r, &mut local_p, &mut local_n, &mut t);
            if local_has_inter {
                has_inter = true;
                if t < *min_t {
                    *min_t = t;
                    *P = local_p.clone();
                    *N = local_n.clone();
                    *sphere_id = i;
                }
            }
        }

        has_inter
    }

}

fn main() {
    const x: i32 = 1920;
    const y: i32 = 1080;
    const fov: f32 = 100.0*PI/180.0;

    let mut image = vec![0u8; (x*y*3) as usize];
    let position_lumiere = Vector { x: 15.0, y: 70.0, z: -30.0};
    let intensite_lumiere = 10000.0;

    let s1 = Sphere { orig: Vector { x: 0.0, y: 0.0, z: -55.0 }, radius: 20.0, albedo: Vector { x: 1.0, y: 0.0, z: 0.0 } };
    let s2 = Sphere { orig: Vector { x: 0.0, y: -2020.0, z: 0.0 }, radius: 2000.0, albedo: Vector { x: 1.0, y: 1.0, z: 1.0 } }; // floor
    let s3 = Sphere { orig: Vector { x: 0.0, y: 2100.0, z: 0.0 }, radius: 2000.0, albedo: Vector { x: 1.0, y: 1.0, z: 1.0 } }; // celling
    let s4 = Sphere { orig: Vector { x: -2050.0, y: 0.0, z: 0.0 }, radius: 2000.0, albedo: Vector { x: 0.0, y: 1.0, z: 0.0 } }; // left wall
    let s5 = Sphere { orig: Vector { x: 2050.0, y: 0.0, z: 0.0 }, radius: 2000.0, albedo: Vector { x: 0.0, y: 0.0, z: 1.0 } }; // right wall
    let s6 = Sphere { orig: Vector { x: 0.0, y: 0.0, z: -2050.0 }, radius: 2000.0, albedo: Vector { x: 0.0, y: 1.0, z: 1.0 } }; // back wall

    let mut scene = Scene { spheres: Vec::new() };
    scene.addSphere(s1);
    scene.addSphere(s2);
    scene.addSphere(s3);
    scene.addSphere(s4);
    scene.addSphere(s5);
    scene.addSphere(s6);

    for i in 0..y {
        for j in 0..x {
            let mut direction = Vector {
                x: j as f32 - x as f32/2.0,
                y: i as f32 - y as f32/2.0,
                z: -x as f32/(2.0*(fov/2.0).tan())
            };
            direction.normalize();
            let r = Ray { orig: Vector { x: 0.0, y: 0.0, z: 0.0 }, dest: direction };

            let mut p = Vector {x: 0.0, y: 0.0, z: 0.0};
            let mut n = Vector {x: 0.0, y: 0.0, z: 0.0};
            let mut intensite_pixel = Vector {x: 0.0, y: 0.0, z: 0.0};
            let mut id: usize = 0;
            let mut t: f32 = 1e99;

            if scene.intersection(&r, &mut p, &mut n, &mut id, &mut t) {

                let mut p_light = Vector {x: 0.0, y: 0.0, z: 0.0};
                let mut n_light = Vector {x: 0.0, y: 0.0, z: 0.0};
                let light_ray = Ray { orig: p.clone() + n.clone()*0.01, dest: (position_lumiere.clone() -p.clone()).getNormalized() };
                let mut id_light: usize = 0;
                let mut t_light: f32 = 1e99;
                let d_light: f32 = (position_lumiere.clone() - p.clone()).getNorm2();
                if scene.intersection(&light_ray, &mut p_light, &mut n_light, &mut id_light, &mut t_light) && t_light*t_light < d_light {
                    intensite_pixel.x = 0.0;
                    intensite_pixel.y = 0.0;
                    intensite_pixel.z = 0.0;
                } else {

                    let lum = intensite_lumiere * 0f32.max( (position_lumiere.clone() - p.clone()).dot(&n) / d_light );
                    intensite_pixel.x = scene.spheres[id].albedo.x * lum;
                    intensite_pixel.y = scene.spheres[id].albedo.y * lum;
                    intensite_pixel.z = scene.spheres[id].albedo.z * lum;

                }

                image[((i*x +j)*3) as usize] = 255f32.min(0f32.max(intensite_pixel.x)) as u8;
                image[((i*x +j)*3 + 1) as usize] = 255f32.min(0f32.max(intensite_pixel.y)) as u8;
                image[((i*x +j)*3 + 2) as usize] = 255f32.min(0f32.max(intensite_pixel.z)) as u8;
            } else {
                image[((i*x +j)*3) as usize] = 0;
                image[((i*x +j)*3 + 1) as usize] = 0;
                image[((i*x +j)*3 + 2) as usize] = 0;
            }
        }
    }

    save_img("./image.bmp", &image, x as u32, y as u32);
}

#[allow(exceeding_bitshifts)]
fn save_img(filename: &str, pixels: &[u8], w: u32, h: u32) {
    let mut bmpfileheader: [u8; 14] = ['B' as u8 ,'M' as u8, 0,0,0,0, 0,0,0,0, 54,0,0,0];
    let mut bmpinfoheader: [u8; 40] = [0; 40];

    bmpinfoheader[0] = 40;
    bmpinfoheader[12] = 1;
    bmpinfoheader[14] = 24;

    let filesize: u32 = 54u32 + w*h*3u32;
    bmpfileheader[2] = filesize as u8;
    bmpfileheader[3] = (filesize >> 8) as u8;
    bmpfileheader[4] = (filesize >> 16) as u8;
    bmpfileheader[5] = (filesize >> 24) as u8;

    bmpinfoheader[4]  = w as u8;
    bmpinfoheader[5]  = (w >> 8) as u8;
    bmpinfoheader[6]  = (w >> 16) as u8;
    bmpinfoheader[7]  = (w >> 24) as u8;
    bmpinfoheader[8]  = h as u8;
    bmpinfoheader[9]  = (h >> 8) as u8;
    bmpinfoheader[10] = (h >> 16) as u8;
    bmpinfoheader[11] = (h >> 24) as u8;

    let mut file = File::create(filename).expect("Unable to write");
    file.write(&bmpfileheader);
    file.write(&bmpinfoheader);

    let mut bgr_pixels = vec![0u8; (h*w*3) as usize];

    for i in 0..w*h {
        bgr_pixels[(i*3) as usize] = pixels[(i*3 + 2) as usize];
        bgr_pixels[(i*3 + 1) as usize] = pixels[(i*3 + 1) as usize];
        bgr_pixels[(i*3 + 2) as usize] = pixels[(i*3) as usize];
    }
    for i in 0..h {
        file.write(&bgr_pixels[(i*w*3) as usize .. ((i+1)*w*3) as usize]);
        let pad_size = (4-(w*3 % 4)) as usize % 4;
        file.write(&vec![0u8; pad_size]);
    }
}
