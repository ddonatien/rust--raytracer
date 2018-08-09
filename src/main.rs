extern crate rand;
extern crate rayon;
extern crate time;

use std::fs::File;
use std::io::prelude::*;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::Mul;
use std::ops::Div;
use std::sync::Arc;
use std::sync::Mutex;
use rayon::prelude::*;
use std::f32::consts::PI;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

#[derive(Debug, Clone)]
struct Vector {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Debug)]
struct Ray {
    orig: Vector,
    dest: Vector,
}

#[derive(Debug)]
struct Sphere {
    orig: Vector,
    radius: f32,
    albedo: Vector,
    miroir: bool,
    transp: bool,
}

struct Scene {
    spheres: Vec<Sphere>,
    lumiere: usize,
    intensite_lumiere: f32,
}

impl Add for Vector {
    type Output = Vector;

    fn add(self, other: Vector) -> Vector {
        Vector { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
}

impl AddAssign for Vector {
    fn add_assign(&mut self, other: Vector) {
        *self = Vector {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        };
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
    fn new() -> Vector {
        Vector {x: 0.0, y: 0.0, z: 0.0}
    }
    fn dot(&self, other: &Vector) -> f32 {
        self.x*other.x + self.y*other.y + self.z*other.z
    }

    fn cross(&self, other: &Vector) -> Vector {
        let crossed = Vector {
            x: self.y*other.z - self.z*other.y,
            y: self.z*other.x - self.x*other.z,
            z: self.x*other.y - self.y*other.x,
        };
        crossed
    }

    fn get_norm2(&self) -> f32 {
        self.dot(&self)
    }

    fn normalize(&mut self) {
        let norm = (self.get_norm2() as f64).sqrt() as f32;
        self.x /= norm;
        self.y /= norm;
        self.z /= norm;
    }

    fn get_normalized(&mut self) -> Vector {
        let mut clone: Vector = self.clone();
        clone.normalize();
        clone
    }

    fn random_cos(&self) -> Vector {
        // Contribution de l'éclairage indirect
        let range = Range::new(0.0, 1.0);
        let mut rng = rand::thread_rng();
        let r1: f32 = range.ind_sample(&mut rng);
        let r2: f32 = range.ind_sample(&mut rng);
        let direction_aleatoire_repere_local = Vector {
            x: (2.0*PI*r1).cos()*(((1.0-r2) as f64).sqrt() as f32),
            y: (2.0*PI*r1).sin()*(((1.0-r2) as f64).sqrt() as f32),
            z: (r2 as f64).sqrt() as f32,
        };
        let aleatoire = Vector {
            x: range.ind_sample(&mut rng),
            y: range.ind_sample(&mut rng),
            z: range.ind_sample(&mut rng),
        };
        let tangent1 = self.clone().cross(&aleatoire);
        let tangent2 = tangent1.cross(&self.clone());
        let direction_aleatoire = self.clone()*direction_aleatoire_repere_local.z
            + tangent1.clone()*direction_aleatoire_repere_local.x
            + tangent2.clone()*direction_aleatoire_repere_local.y;
        direction_aleatoire
    }
}

impl Sphere {

    fn intersection(&self, r: &Ray, p: &mut Vector, n: &mut Vector, t: &mut f32) -> bool {
        // résoud a*t^2 + b*t + c = 0

        let diff: Vector = r.orig.clone() - self.orig.clone();
        let a: f32 = 1.0;
        let b: f32 = 2.0 * r.dest.clone().dot(&diff);
        let c: f32 = diff.get_norm2() - self.radius*self.radius;

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

                let _p = r.orig.clone() + r.dest.clone()* *t;
                p.x = _p.x;
                p.y = _p.y;
                p.z = _p.z;
                let normale = (_p - self.orig.clone()).get_normalized();
                n.x = normale.x;
                n.y = normale.y;
                n.z = normale.z;
                true
            }
        }

    }

}

impl Scene {

    fn add_sphere(&mut self, sphere: Sphere) {
        self.spheres.push(sphere);
    }

    fn intersection(&self, r: &Ray, p: &mut Vector, n: &mut Vector, sphere_id: &mut usize, min_t: &mut f32) -> bool {
        let mut has_inter: bool = false;
        *min_t = 1e10;

        for (i, s) in self.spheres.iter().enumerate() {
            let mut t: f32 = 0.0;
            let mut local_p: Vector = Vector {x: 0.0, y: 0.0, z: 0.0};
            let mut local_n: Vector = Vector {x: 0.0, y: 0.0, z: 0.0};
            let local_has_inter: bool = s.intersection(r, &mut local_p, &mut local_n, &mut t);
            if local_has_inter {
                has_inter = true;
                if t < *min_t {
                    *min_t = t;
                    *p = local_p.clone();
                    *n = local_n.clone();
                    *sphere_id = i;
                }
            }
        }

        has_inter
    }

}

fn main() {
    let time_start = time::now().tm_sec;
    let time_start_min = time::now().tm_min;
    const X: i32 = 800;
    const Y: i32 = 800;
    const NRAYS: u16 = 600;
    const FOV: f32 = 60.0*PI/180.0;

    let mut image = vec![0u8; (X*Y*3) as usize];
    let position_lumiere = Vector { x: 15.0, y: 70.0, z: -30.0};
    let intensite_lumiere = 1000000000.0 * 4.0 * PI / ( 4.0 * PI * 30.0 * 30.0 * PI);
    let position_camera = Vector::new();
    let focus_distance = 60.0;

    eprintln!("Création des paramètres : {:?}", time::now().tm_sec - time_start);
    let slum = Sphere { orig: position_lumiere, radius: 15.0, albedo: Vector { x: 1.0, y: 1.0, z: 1.0 }, miroir: false, transp: false };
    let s1 = Sphere { orig: Vector { x: 0.0, y: 0.0, z: -55.0 }, radius: 10.0, albedo: Vector { x: 1.0, y: 0.0, z: 1.0 }, miroir: false, transp: false };
    let s1bis = Sphere { orig: Vector { x: -15.0, y: 0.0, z: -35.0 }, radius: 10.0, albedo: Vector { x: 1.0, y: 0.0, z: 0.0 }, miroir: false, transp: true };
    let s1ter = Sphere { orig: Vector { x: 15.0, y: 0.0, z: -75.0 }, radius: 10.0, albedo: Vector { x: 1.0, y: 0.0, z: 0.0 }, miroir: true, transp: false };
    let s2 = Sphere { orig: Vector { x: 0.0, y: -2020.0, z: 0.0 }, radius: 2000.0, albedo: Vector { x: 1.0, y: 1.0, z: 1.0 }, miroir: false, transp: false }; // floor
    let s3 = Sphere { orig: Vector { x: 0.0, y: 2100.0, z: 0.0 }, radius: 2000.0, albedo: Vector { x: 1.0, y: 1.0, z: 1.0 }, miroir: false, transp: false }; // celling
    let s4 = Sphere { orig: Vector { x: -2050.0, y: 0.0, z: 0.0 }, radius: 2000.0, albedo: Vector { x: 0.0, y: 1.0, z: 0.0 }, miroir: false, transp: false }; // left wall
    let s5 = Sphere { orig: Vector { x: 2050.0, y: 0.0, z: 0.0 }, radius: 2000.0, albedo: Vector { x: 0.0, y: 0.0, z: 1.0 }, miroir: false, transp: false }; // right wall
    let s6 = Sphere { orig: Vector { x: 0.0, y: 0.0, z: -2100.0 }, radius: 2000.0, albedo: Vector { x: 0.0, y: 1.0, z: 1.0 }, miroir: false, transp: false }; // back wall

    let mut scene = Scene { spheres: Vec::new(), lumiere: 0, intensite_lumiere: intensite_lumiere };
    scene.add_sphere(slum);
    scene.add_sphere(s1);
    scene.add_sphere(s1bis);
    scene.add_sphere(s1ter);
    scene.add_sphere(s2);
    scene.add_sphere(s3);
    scene.add_sphere(s4);
    scene.add_sphere(s5);
    scene.add_sphere(s6);
    eprintln!("Création de la scène : {:?}", time::now().tm_sec - time_start);
    let arc_image = Arc::new(Mutex::new(image));

    for i in 0..Y {
            (0..X).into_par_iter()
                .for_each(|j| fill_image(i, j, &scene, NRAYS, X, Y, FOV, &position_camera, focus_distance, Arc::clone(&arc_image)));
    }

    eprintln!("Fin du calcul de l'image {:?}", time::now().tm_min - time_start_min);

    let image = arc_image.lock().unwrap();
    save_img("./image.bmp", &image, X as u32, Y as u32);
}

fn fill_image( i: i32, j: i32, scene: &Scene, n_rays: u16, x: i32, y: i32, fov: f32, position_camera: &Vector, focus_distance: f32, image: Arc<Mutex<Vec<u8>>>) {
    let mut color = Vector::new();
    for n in 0..n_rays {

        // Methode de box Muller
        let range = Range::new(0.0, 1.0);
        let mut rng = rand::thread_rng();
        let r1: f32 = range.ind_sample(&mut rng);
        let r2: f32 = range.ind_sample(&mut rng);

        let r = (-2.0*r1.log(10.0) as f64).sqrt() as f32;
        let dx = r*(2.0*PI*r2).cos();
        let dy = r*(2.0*PI*r2).sin();

        let dx_aperture = (range.ind_sample(&mut rng) - 0.5) * 5.0;
        let dy_aperture = (range.ind_sample(&mut rng) - 0.5) * 5.0;

        let mut direction = Vector {
            x: j as f32 - x as f32/2.0 + 0.5 + dx,
            y: i as f32 - y as f32/2.0 + 0.5 + dy,
            z: -x as f32/(2.0*(fov/2.0).tan())
        };
        direction.normalize();


        let destination = position_camera.clone() + direction * focus_distance;
        let new_origin = position_camera.clone() + Vector { x: dx_aperture, y: dy_aperture, z: 0.0};
        let r = Ray { orig: new_origin.clone(), dest: (destination.clone() - new_origin.clone()).get_normalized() };

        color += get_color( &r, &scene, 5, true )/(n_rays as f32);
    }

    {
        let mut image = image.lock().unwrap();
        image[((i*x + j)*3) as usize] = 255f32.min(0f32.max((color.x).powf((1.0)/(2.2)))) as u8;
        image[((i*x + j)*3 + 1) as usize] = 255f32.min(0f32.max((color.y).powf((1.0)/(2.2)))) as u8;
        image[((i*x + j)*3 + 2) as usize] = 255f32.min(0f32.max((color.z).powf((1.0)/(2.2)))) as u8;
    }

}

fn get_color(r: &Ray, scene: &Scene, nbrebonds: u8, show_lights: bool) -> Vector {

    if nbrebonds == 0 {
        let intensite_pixel = Vector {x: 0.0, y: 0.0, z: 0.0};
        return intensite_pixel;
    }

    let mut p = Vector {x: 0.0, y: 0.0, z: 0.0};
    let mut n = Vector {x: 0.0, y: 0.0, z: 0.0};
    let mut intensite_pixel = Vector {x: 0.0, y: 0.0, z: 0.0};
    let mut id: usize = 0;
    let mut t: f32 = 1e10;

    if scene.intersection(&r, &mut p, &mut n, &mut id, &mut t) {

        if id == scene.lumiere {
            let intensite_pixel = if show_lights { scene.spheres[scene.lumiere].albedo.clone()*scene.intensite_lumiere } else { Vector::new() };
            return intensite_pixel;
        } else {

            if scene.spheres[id].miroir {

                let direction_miroir = r.dest.clone() - n.clone()*n.clone().dot(&r.dest.clone())*2.0;
                let rayon_miroir = Ray { orig: p + n*0.001, dest: direction_miroir };
                intensite_pixel = get_color( &rayon_miroir, &scene, nbrebonds -1, show_lights );
                return intensite_pixel;

            } else {

                if scene.spheres[id].transp {

                    let mut n1 = 0.0;
                    let mut n2 = 0.0;
                    let mut normale_pour_transparence = Vector::new();

                    if r.dest.dot(&n) > 0.0 {

                        n1 = 1.3;
                        n2 = 1.0;
                        normale_pour_transparence = Vector { x: 0.0, y: 0.0, z: 0.0 } - n.clone();

                    } else {

                        n1 = 1.0;
                        n2 = 1.3;
                        normale_pour_transparence = n.clone();

                    }

                    let lhs = (n1/n2)*(n1/n2);
                    let rhs = 1.0 - normale_pour_transparence.clone().dot(&r.dest)*normale_pour_transparence.clone().dot(&r.dest);
                    let radical = 1.0 - lhs*rhs;

                    if radical > 0.0 {

                        let direction_refracte = (r.dest.clone() - normale_pour_transparence.clone()*(r.dest.dot(&normale_pour_transparence.clone())))*(n1/n2) - normale_pour_transparence.clone()*((radical as f64).sqrt() as f32);
                        let rayon_refracte = Ray { orig: p - normale_pour_transparence.clone()*0.001, dest: direction_refracte };
                        intensite_pixel = get_color( &rayon_refracte, &scene, nbrebonds -1, show_lights );
                        return intensite_pixel;

                    } else {

                        intensite_pixel.x = 0.0;
                        intensite_pixel.y = 0.0;
                        intensite_pixel.z = 0.0;

                        return intensite_pixel;

                    }

                } else {

                    let axe_op = (p.clone() - scene.spheres[scene.lumiere].orig.clone()).get_normalized();
                    let dir_aleatoire = axe_op.random_cos();
                    let point_aleatoire = dir_aleatoire.clone() * scene.spheres[scene.lumiere].radius + scene.spheres[scene.lumiere].orig.clone() ;
                    let wi = (point_aleatoire.clone() - p.clone()).get_normalized();
                    let d_light2 = (point_aleatoire.clone() - p.clone()).get_norm2();
                    let np = dir_aleatoire.clone();

                    let mut p_light = Vector::new();
                    let mut n_light = Vector::new();
                    let light_ray = Ray { orig: p.clone() + n.clone()*0.01, dest: wi.clone() };
                    let mut id_light: usize = 0;
                    let mut t_light: f32 = 1e10;

                    if scene.intersection(&light_ray, &mut p_light, &mut n_light, &mut id_light, &mut t_light) && t_light*t_light < d_light2*0.99 {

                        intensite_pixel = Vector::new();

                    } else {

                        let brdf = scene.spheres[id].albedo.clone() / PI;
                        let proba = axe_op.dot(&dir_aleatoire) / ( PI * scene.spheres[scene.lumiere].radius * scene.spheres[scene.lumiere].radius);
                        let j = 1.0 * np.dot(&(Vector::new() - wi.clone())) / d_light2;
                        let intensite_pixel =  brdf * scene.intensite_lumiere * 0f32.max(n.clone().dot(&wi.clone())) * j / proba;

                    }


                    // Contribution de l'éclairage indirect
                    let direction_aleatoire = n.random_cos();
                    let rayon_aleatoire = Ray { orig: p + n*0.001, dest: direction_aleatoire };
                    let albedo_local = scene.spheres[id].albedo.clone();
                    let color = get_color( &rayon_aleatoire, &scene, nbrebonds -1, true )*albedo_local;
                    intensite_pixel += color;
                    return intensite_pixel;
                }
            }
        }
    } else {

        intensite_pixel.x = 0.0;
        intensite_pixel.y = 0.0;
        intensite_pixel.z = 0.0;

        return intensite_pixel;

    }

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
