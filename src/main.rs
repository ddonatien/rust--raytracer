extern crate rand;
extern crate rayon;
extern crate time;

use std::fs::File;
use std::io::prelude::*;
use std::ops::AddAssign;
use std::sync::Arc;
use std::sync::Mutex;
use rayon::prelude::*;
use std::f32::consts::PI;
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};

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

#[derive(Debug)]
struct Triangle {
    A: Vector,
    B: Vector,
    C: Vector,
    albedo: Vector,
    miroir: bool,
    transp: bool,
}

trait Shape {
    fn intersection(&self, r: &Ray, p: &mut Vector, n: &mut Vector, t: &mut f32) -> bool;

    fn radius(&self) -> f32;

    fn albedo(&self) -> Vector;

    fn orig(&self) -> Vector;

    fn transp(&self) -> bool;

    fn miroir(&self) -> bool;
}

struct Scene {
    shapes: Vec<Box<Shape + Sync>>,
    lumiere: usize,
    intensite_lumiere: f32,
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

impl Vector {
    fn new() -> Vector {
        Vector {x: 0.0, y: 0.0, z: 0.0}
    }

    fn add(&self, other: &Vector) -> Vector {
        Vector { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    fn sub(&self, other: &Vector) -> Vector {
        Vector { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    fn fmul(&self, scalar: f32) -> Vector {
        Vector { x: scalar*self.x, y: scalar*self.y, z: scalar*self.z}
    }

    fn mul(&self, other: &Vector) -> Vector {
        Vector { x: other.x*self.x, y: other.y*self.y, z: other.z*self.z}
    }

    fn div(&self, scalar: f32) -> Vector {
        Vector { x: self.x/scalar, y: self.y/scalar, z: self.z/scalar}
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
        let range = Uniform::new(0.0, 1.0);
        // let mut rng = rand::thread_rng();
        let r1: f32 = thread_rng().sample(range);
        let r2: f32 = thread_rng().sample(range);
        let direction_aleatoire_repere_local = Vector {
            x: (2.0*PI*r1).cos()*(((1.0-r2) as f64).sqrt() as f32),
            y: (2.0*PI*r1).sin()*(((1.0-r2) as f64).sqrt() as f32),
            z: (r2 as f64).sqrt() as f32,
        };
        let aleatoire = Vector {
            x: thread_rng().sample(range),
            y: thread_rng().sample(range),
            z: thread_rng().sample(range),
        };
        let tangent1 = self.clone().cross(&aleatoire);
        let tangent2 = tangent1.cross(&self.clone());
        let direction_aleatoire = self.fmul(direction_aleatoire_repere_local.z)
            .add(&tangent1.fmul(direction_aleatoire_repere_local.x))
            .add(&tangent2.fmul(direction_aleatoire_repere_local.y));
        direction_aleatoire
    }
}

impl Shape for Sphere {
    fn intersection(&self, r: &Ray, p: &mut Vector, n: &mut Vector, t: &mut f32) -> bool {
        // résoud a*t^2 + b*t + c = 0

        let diff: Vector = r.orig.sub(&self.orig);
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

                let _p = r.orig.add(&r.dest.fmul(*t));
                p.x = _p.x;
                p.y = _p.y;
                p.z = _p.z;
                let normale = _p.sub(&self.orig).get_normalized();
                n.x = normale.x;
                n.y = normale.y;
                n.z = normale.z;
                true
            }
        }

    }

    fn radius(&self) -> f32 {
        self.radius
    }

    fn albedo(&self) -> Vector {
        self.albedo.clone()
    }

    fn orig(&self) -> Vector {
        self.orig.clone()
    }

    fn transp(&self) -> bool {
        self.transp
    }

    fn miroir(&self) ->  bool {
        self.miroir
    }
}

impl Shape for Triangle {
    fn intersection(&self, r: &Ray, p: &mut Vector, n: &mut Vector, t: &mut f32) -> bool {
        let n_tmp = self.B.sub(&self.A).cross(&self.C.sub(&self.A)).get_normalized();
        let t_tmp = (self.C.sub(&r.orig)).dot(&n_tmp.clone()) / r.dest.clone().dot(&n_tmp.clone());
        if t_tmp < 0.0 {
            return false;
        }
        let p_tmp = r.orig.add(&r.dest.fmul(t_tmp));
        let u = self.B.sub(&self.A);
        let v = self.C.sub(&self.A);
        let w = p_tmp.sub(&self.A);

        let m11 = u.get_norm2();
        let m12 = u.clone().dot(&v.clone());
        let m22 = v.get_norm2();
        let detm = m11 * m22 - m12 * m12; // Matrice symétrique

        let b11 = w.clone().dot(&u.clone());
        let b21 = w.clone().dot(&v.clone());
        let detb = b11 * m22 - b21 * m12;
        let beta = detb / detm;

        if beta < 0. || beta > 1. {
            return false;
        }

        let g12 = b11;
        let g22 = b21;
        let detg = m11 * g22 - g12 * m12;
        let gamma = detg / detm;

        if gamma < 0. || gamma > 1. {
            return false;
        }

        let alpha = 1. - beta - gamma;

        if alpha < 0. || alpha > 1. {
            return false;
        }
        if alpha + beta + gamma > 1. {
            return false;
        }

        *t = t_tmp;

        p.x = p_tmp.x;
        p.y = p_tmp.y;
        p.z = p_tmp.z;

        n.x = n_tmp.x;
        n.y = n_tmp.y;
        n.z = n_tmp.z;
        return true;
    }

    fn radius(&self) -> f32 {
        0.
    }

    fn albedo(&self) -> Vector {
        self.albedo.clone()
    }

    fn orig(&self) -> Vector {
        Vector::new()
    }

    fn transp(&self) -> bool {
        self.transp
    }

    fn miroir(&self) ->  bool {
        self.miroir
    }
}

impl Scene {

    fn add_sphere(&mut self, sphere: Sphere) {
        self.shapes.push(Box::new(sphere));
    }

    fn add_triangle(&mut self, triangle: Triangle) {
        self.shapes.push(Box::new(triangle));
    }

    fn intersection(&self, r: &Ray, p: &mut Vector, n: &mut Vector, id: &mut usize, min_t: &mut f32) -> bool {
        let mut has_inter: bool = false;
        *min_t = 1e10;

        for (i, s) in self.shapes.iter().enumerate() {
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
                    *id = i;
                }
            }
        }

        has_inter
    }

}

fn main() {

    // Scene global settings
    const X: i32 = 800;
    const Y: i32 = 800;
    const NRAYS: u16 = 600;
    const NBOUNCES: u8 = 3;
    let focus_distance = 60.0;
    const FOV: f32 = 60.0*PI/180.0;
    let mut image = vec![0u8; (X*Y*3) as usize];

    // Camera an light settings
    let position_lumiere = Vector { x: 15.0, y: 70.0, z: -30.0};
    let intensite_lumiere = 1000000000.0 * 4.0 * PI / ( 4.0 * PI * 30.0 * 30.0 * PI);
    let position_camera = Vector::new();


    // Scene objects creation
    let slum = Sphere { orig: position_lumiere, radius: 15.0, albedo: Vector { x: 1.0, y: 1.0, z: 1.0 }, miroir: false, transp: false };
    let s1 = Sphere { orig: Vector { x: 0.0, y: 0.0, z: -55.0 }, radius: 10.0, albedo: Vector { x: 1.0, y: 0.0, z: 1.0 }, miroir: false, transp: false };
    let s1bis = Sphere { orig: Vector { x: -15.0, y: 0.0, z: -35.0 }, radius: 10.0, albedo: Vector { x: 1.0, y: 0.0, z: 0.0 }, miroir: false, transp: true };
    let s1ter = Sphere { orig: Vector { x: 15.0, y: 0.0, z: -75.0 }, radius: 10.0, albedo: Vector { x: 1.0, y: 0.0, z: 0.0 }, miroir: true, transp: false };
    let s2 = Sphere { orig: Vector { x: 0.0, y: -2020.0, z: 0.0 }, radius: 2000.0, albedo: Vector { x: 1.0, y: 1.0, z: 1.0 }, miroir: false, transp: false }; // floor
    let s3 = Sphere { orig: Vector { x: 0.0, y: 2100.0, z: 0.0 }, radius: 2000.0, albedo: Vector { x: 1.0, y: 1.0, z: 1.0 }, miroir: false, transp: false }; // celling
    let s4 = Sphere { orig: Vector { x: -2050.0, y: 0.0, z: 0.0 }, radius: 2000.0, albedo: Vector { x: 0.0, y: 1.0, z: 0.0 }, miroir: false, transp: false }; // left wall
    let s5 = Sphere { orig: Vector { x: 2050.0, y: 0.0, z: 0.0 }, radius: 2000.0, albedo: Vector { x: 0.0, y: 0.0, z: 1.0 }, miroir: false, transp: false }; // right wall
    let s6 = Sphere { orig: Vector { x: 0.0, y: 0.0, z: -2100.0 }, radius: 2000.0, albedo: Vector { x: 0.0, y: 1.0, z: 1.0 }, miroir: false, transp: false }; // back wall
    let tri = Triangle { A: Vector { x: -10., y: -10., z: -35.}, B: Vector { x: 10., y: -10., z: -80.}, C: Vector { x: 0., y: 10., z: -80.}, albedo: Vector {x: 0., y: 1., z: 0.}, miroir: false, transp: false };

    let mut scene = Scene { shapes: Vec::new(), lumiere: 0, intensite_lumiere: intensite_lumiere };
    scene.add_sphere(slum);

    scene.add_triangle(tri);
    scene.add_sphere(s1);
    scene.add_sphere(s1bis);
    scene.add_sphere(s1ter);
    scene.add_sphere(s2);
    scene.add_sphere(s3);
    scene.add_sphere(s4);
    scene.add_sphere(s5);
    scene.add_sphere(s6);

    let arc_image = Arc::new(Mutex::new(image)); // Image is stored is an arc mutex for mutli-threaded rendering

    // Redering pixel by pixel
    for i in 0..Y {
            (0..X).into_par_iter()
                .for_each(|j| fill_image(i, j, &scene, NRAYS, NBOUNCES, X, Y, FOV, &position_camera, focus_distance, Arc::clone(&arc_image)));
    }

    // Saving result
    let image = arc_image.lock().unwrap();
    save_img("./image.bmp", &image, X as u32, Y as u32);
}

fn fill_image( i: i32, j: i32, scene: &Scene, n_rays: u16, n_rebonds: u8, x: i32, y: i32, fov: f32, position_camera: &Vector, focus_distance: f32, image: Arc<Mutex<Vec<u8>>>) {

    let mut color = Vector::new();

    for _n in 0..n_rays {

        // Box-Muller method

        let range = Uniform::new(0.0, 1.0);
        // let mut rng = rand::thread_rng();
        let r1: f32 = thread_rng().sample(range);
        let r2: f32 = thread_rng().sample(range);

        // let r = (-2.0*r1.log(10.0) as f64).sqrt() as f32;
        let r = (-2.0*r1.ln() as f64).sqrt() as f32; // I used log10, but it seems Box-Muller method uses ln
        let dx = r*(2.0*PI*r2).cos();
        let dy = r*(2.0*PI*r2).sin();

        let dx_aperture = (thread_rng().sample(range) - 0.5) * 5.0;
        let dy_aperture = (thread_rng().sample(range) - 0.5) * 5.0;

        let mut direction = Vector {
            x: j as f32 - x as f32/2.0 + 0.5 + dx,
            y: i as f32 - y as f32/2.0 + 0.5 + dy,
            z: -x as f32/(2.0*(fov/2.0).tan())
        };
        direction.normalize();


        let destination = position_camera.add(&direction.fmul(focus_distance));
        let new_origin = position_camera.add(&Vector { x: dx_aperture, y: dy_aperture, z: 0.0});
        let r = Ray { orig: new_origin.clone(), dest: (destination.sub(&new_origin)).get_normalized() };

        color += get_color( &r, &scene, n_rebonds, true ).div(n_rays as f32);
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
            let intensite_pixel = if show_lights { scene.shapes[scene.lumiere].albedo().fmul(scene.intensite_lumiere) } else { Vector::new() };
            return intensite_pixel;
        } else {

            if scene.shapes[id].miroir() {

                let direction_miroir = r.dest.sub(&n.fmul(n.dot(&r.dest)*2.0));
                let rayon_miroir = Ray { orig: p.add(&n.fmul(0.001)), dest: direction_miroir };
                intensite_pixel = get_color( &rayon_miroir, &scene, nbrebonds - 1, show_lights );
                return intensite_pixel;

            } else {

                if scene.shapes[id].transp() {

                    let mut n1 = 0.0;
                    let mut n2 = 0.0;
                    let mut normale_pour_transparence = Vector::new();

                    if r.dest.dot(&n) > 0.0 {

                        n1 = 1.3;
                        n2 = 1.0;
                        normale_pour_transparence = (Vector { x: 0.0, y: 0.0, z: 0.0 }).sub(&n);

                    } else {

                        n1 = 1.0;
                        n2 = 1.3;
                        normale_pour_transparence = n.clone();

                    }

                    let lhs = (n1/n2)*(n1/n2);
                    let rhs = 1.0 - normale_pour_transparence.dot(&r.dest) * normale_pour_transparence.dot(&r.dest);
                    let radical = 1.0 - lhs*rhs;

                    if radical > 0.0 {

                        let direction_refracte = ((r.dest.clone().sub(&normale_pour_transparence.fmul(r.dest.dot(&normale_pour_transparence)))).fmul(n1/n2)).sub(&normale_pour_transparence.fmul((radical as f64).sqrt() as f32));
                        let rayon_refracte = Ray { orig: p.sub(&normale_pour_transparence.fmul(0.001)), dest: direction_refracte };
                        intensite_pixel = get_color( &rayon_refracte, &scene, nbrebonds -1, show_lights );
                        return intensite_pixel;

                    } else {

                        intensite_pixel.x = 0.0;
                        intensite_pixel.y = 0.0;
                        intensite_pixel.z = 0.0;

                        return intensite_pixel;

                    }

                } else {

                    let axe_op = (p.sub(&scene.shapes[scene.lumiere].orig())).get_normalized();
                    let dir_aleatoire = axe_op.random_cos();
                    let point_aleatoire = dir_aleatoire.fmul(scene.shapes[scene.lumiere].radius()).add(&scene.shapes[scene.lumiere].orig()) ;
                    let wi = (point_aleatoire.sub(&p)).get_normalized();
                    let d_light2 = (point_aleatoire.sub(&p)).get_norm2();
                    let np = dir_aleatoire.clone();

                    let mut p_light = Vector::new();
                    let mut n_light = Vector::new();
                    let light_ray = Ray { orig: p.add(&n.fmul(0.01)), dest: wi.clone() };
                    let mut id_light: usize = 0;
                    let mut t_light: f32 = 1e10;

                    if scene.intersection(&light_ray, &mut p_light, &mut n_light, &mut id_light, &mut t_light) && t_light*t_light < d_light2*0.99 {

                        intensite_pixel = Vector::new();

                    } else {

                        let brdf = scene.shapes[id].albedo().div(PI);
                        let proba = axe_op.dot(&dir_aleatoire) / ( PI * scene.shapes[scene.lumiere].radius() * scene.shapes[scene.lumiere].radius());
                        let j = 1.0 * np.dot(&(Vector::new().sub(&wi))) / d_light2;
                        let intensite_pixel =  brdf.fmul(scene.intensite_lumiere * 0f32.max(n.dot(&wi)) * j / proba);

                    }


                    // Contribution de l'éclairage indirect
                    let direction_aleatoire = n.random_cos();
                    let rayon_aleatoire = Ray { orig: p.add(&n.fmul(0.001)), dest: direction_aleatoire };
                    let albedo_local = &scene.shapes[id].albedo();
                    let color = get_color( &rayon_aleatoire, &scene, nbrebonds -1, true ).mul(albedo_local);
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
