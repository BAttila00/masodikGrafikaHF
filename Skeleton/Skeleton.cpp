
#include "framework.h"

const char* vertexSourceForTexturing = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
																				//most igy adjuk meg az uv vektort, de lehetne normal modon is
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSourceForTexturing = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

//------------------------------------------------

const vec3 La = vec3(0.5f, 0.6f, 0.6f);
const vec3 Le = vec3(0.8f, 0.8f, 0.8f);
const vec3 lightPosition = vec3(0.4f, 0.4f, 0.25f);
const vec3 ka = vec3(0.5f, 0.5f, 0.5f);
const float shininess = 500.0f;
const int maxdepth = 5;
const float epsilon = 0.01f;		//eredeti: 0.01

struct Hit {
	float t;		//a metsz�s ideje
	vec3 position;	//a metsz�spont helye
	vec3 normal;	//a metsz�spontban vett norm�lvektor
	int mat;	// a metsz�spontban l�v� material id-ja
};

struct Ray {
	vec3 start, dir;
	vec3 weight;

	Ray(vec3 _start, vec3 _dir) {
		start = _start;
		dir = normalize(_dir);
		weight = vec3(1, 1, 1);
	}
};

const int objFaces = 12;

vec3 kd[2], ks[2], F0Gold;
vec3 F0PerfectReflect;

class ConvexPolyhedron {
	vec3 v[20];
	int planes[objFaces * 3];
public:
	ConvexPolyhedron() {

	};
	ConvexPolyhedron(std::vector<vec3> _v, std::vector<int> _planes) {
		for (size_t i = 0; i < _v.size(); i++)
		{
			v[i] = _v[i];
		}
		for (size_t i = 0; i < _planes.size(); i++)
		{
			planes[i] = _planes[i];
		}
	}

	/// <summary>
	/// Meghat�roz egy s�kot egy pontj�val �s norm�lvektor�val
	/// </summary>
	/// <param name="i"></param>
	/// <param name="scale"></param>
	/// <param name="p">Kimen� param�ter: a s�k egy pontja</param>
	/// <param name="normal">Kimen� param�ter: a s�k norm�lvektora</param>
	void getObjPlane(int i, float scale, vec3& p, vec3& normal) {
		vec3 p1 = v[planes[3 * i] - 1];
		vec3 p2 = v[planes[3 * i + 1] - 1];
		vec3 p3 = v[planes[3 * i + 2] - 1];
		normal = cross(p2 - p1, p3 - p1);
		if (dot(p1, normal) < 0) normal = -normal;
		p = p1 * scale + vec3(0, 0, 0.03f);
	}

	Hit intersectConvexPolyhedron(Ray ray, Hit hit, float scale, int mat) {
		for (int i = 0; i < objFaces; i++)
		{
			vec3 p1;
			vec3 normal;
			getObjPlane(i, scale, p1, normal);
			float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
			if (ti <= epsilon || (ti > hit.t && hit.t > 0)) continue;
			vec3 intersect = ray.start + ray.dir * ti;
			bool outside = false;
			bool closeToEdge = false;
			for (int j = 0; j < objFaces; j++) {
				if (i == j) continue;
				vec3 p11, n;
				getObjPlane(j, scale, p11, n);
				//printf("%f; ", dot(n, intersect - p11));
				if (dot(n, intersect - p11) > 0) {
					outside = true;
					break;
				}
				if (dot(n, intersect - p11) > -0.15f && dot(n, intersect - p11) <= 0.0f) {
					hit.mat = 0;				//ha k�zel vagyunk a dedoka�der sz�leihez akkor legyen a material 0-�s azonos�t�j�, azaz diff�z sz�rk�s
					closeToEdge = true;
				}
			}
			if (!outside) {
				hit.t = ti;
				hit.position = intersect;
				hit.normal = normalize(normal);
				if(!closeToEdge)
					hit.mat = mat;
			}
		}
		return hit;
	}


};

ConvexPolyhedron convexPolyhedron;

class Sphere {		//8. dia
	vec3 center;
	float radius;
public:
	Sphere() {

	}

	Sphere(const vec3& _center, float _radius) {
		center = _center; 
		radius = _radius;
	}

	Hit intersect(const Ray& ray, int material) {		//a kapott sug�r hol metszi el ezt a g�mb�t
		Hit hit;
		hit.t = -1;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4 * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2 / a;
		float t2 = (-b - sqrt_discr) / 2 / a;
		if (t1 <= 0) return hit; // t1 >= t2 for sure
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) / radius;		//g�mb eset�n �gy sz�m�thatjuk ki a norm�lvektort adott pontban (de csak g�mbn�l)
		hit.mat = material;
		return hit;
	}
};

class ImplicitSurface {
	float a, b, c;
	//a fel�let implicit egyenlete: exp(a*x^2+b*y^2-cz)-1 = 0 (a,b,c pozit�v nem eg�sz) ->
	// -> exp(a*x^2+b*y^2-cz) = 1 -> a*x^2+b*y^2-cz = 0 egyenletet kell megoldani

public:
	ImplicitSurface() {

	}
	ImplicitSurface(float _a, float _b, float _c) {
		a = _a;
		b = _b;
		c = _c;
	}

	Hit intersect(const Ray& ray, int material) {
		//a fel�let implicit egyenlet�be behelyettes�tj�k a sug�r egyenlet�t
		//pontosabban ebbe: mert el�g ha itt teljes�l az egyenl� nulla felt�tel: a*x^2+b*y^2-cz = 0
		//a* (ray.start.x + ray.dir.x * t_metszes) ^ 2 + b * (ray.start.y + ray.dir.y * t_metszes) ^ 2 - c * (ray.start.z + ray.dir.z * t_metszes) = 0
		//fenti m�sodfok� egyenletben t_metszes az ismeretlen (azon t id�pillanat amikor a sug�r elmetszi ezen fel�letet)
		//megoldani t_metszes-re, majd a kisebbik (de nagyobb nulla) t_metszes-t visszahelyettes�teni: hit.position = ray.start + ray.dir * t_metszes egyenletbe
		//ebb�l kapunk egymetsz�spontot,amire meg kell n�zni h belef�r-e a megadott k�rbe -> 0,0,0 k�z�ppont� k�rt�l 0.3-n�l kisebb t�vols�gra van-e
		//ha nem akkor eldobjuk, azaz hit.t = -1-et �ll�tjuk be

		Hit hit;
		hit.t = -1;			//ha t kisebb mint 0 akkor a sug�r nem metszi el ezt az objektumot
		float t_metszes;	//a metsz�s id�pontja
		float apar, bpar, cpar;		//a m�sodfok� egyenlet megold�k�plet�nek egy�tthat�i
		apar = a * pow(ray.dir.x, 2) + b * pow(ray.dir.y, 2);
		bpar = a * 2 * ray.start.x * ray.dir.x + b * 2 * ray.start.y * ray.dir.y + c * ray.dir.z;
		cpar = a * pow(ray.start.x, 2) + b * pow(ray.start.y, 2) + c * ray.start.z;
		if (apar == 0) printf("apar is zero");
		float discr = bpar * bpar - 4 * apar * cpar;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-bpar + sqrt_discr) / 2 / apar;
		float t2 = (-bpar - sqrt_discr) / 2 / apar;
		if (t1 <= 0) return hit;	//ha t kisebb mint nulla akkor a sug�r nem metszi el ezt az objektumot
									//mivel t1 nagyobb mint t2 �gy t2 is biztos negat�v
		t_metszes = (t2 > 0) ? t2 : t1;		//ha t2 nagyobb mint nulla akkor mindig v�lasszuk azt hiszen biztos h kisebb mint t1
		vec3 position = ray.start + ray.dir * t_metszes;		//a metsz�s helye
		vec3 distVector = position - vec3(0.0f, 0.0f, 0.0f);	//meg kell n�zn�nk h ezen metsz�shely a 0.3 sugar� 0,0,0 k�z�ppont� k�r�n bel�l van-e
		float dist = length(distVector);
		if (dist > 0.3) return hit;								//ha nincs bel�l akkor dobjuk el a metsz�spontot -> nincs metsz�spont
		hit.t = t_metszes;										//egy�bk�nt a norm�l m�don j�rjunk el, �ll�tsuk be a hit (metsz�s) tulajdons�gait �s t�rj�nk vissza vele
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(gradiens(position));
		hit.mat = material;
		return hit;
	}

	/// <summary>
	/// Gradiens kisz�m�t�sa adott pontban (Norm�lvektor sz�m�t�s�hoz)
	/// </summary>
	/// <param name="position"></param>
	/// <returns>a norm�lvektor az adott pontban</returns>
	vec3 gradiens(vec3 position) {
		//https://www.derivative-calculator.net/
		//e^(ax^2+by^2-cz)-1 egyenletet kell lederiv�lni x, y, majd z szerint
		//majd az deriv�ltba behelyettes�teni a position-t (azaz x hely�re postion.x-et, y hely�re postion.y-t, z hely�re postion.z-t)
		//gradiens sz�mol�s: https://sze-gyor.videotorium.hu/hu/recordings/37079/gradiens-vektor-1
		float dx = 2 * a * position.x * exp(a * pow(position.x, 2) - c * position.z + b * pow(position.y, 2));
		float dy = 2 * b * position.y * exp(b * pow(position.y, 2) - c * position.z + a * pow(position.x, 2));
		float dz = -c * exp(-c * position.z + b * pow(position.y, 2) + a * pow(position.x, 2));
		return vec3(dx, dy, dz);
	}
};

vec3 reflect(vec3 V, vec3 N) {
	return V - N * dot(N, V) * 2;
};

//------------------------------------------------

class Camera {
	vec3 lookat, right;
	vec3 pvup;	//prefer�lt f�gg�leges ir�ny
	vec3 rvup;	//val�di f�gg�leges ir�ny
	float fov = 45 * (float)M_PI / 180;
public:
	vec3 eye;
	Camera() : eye(1, 0, 0), pvup(0, 0, -1), lookat(0, 0, 0) {		//Camera() : eye(1, 0, 0), pvup(0, 0, 1), lookat(0, 0, 0) {        -> ez is j�l mutat
		set();
	}
	void set() {
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(pvup, w)) * f * tanf(fov / 2);
		rvup = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	void Animate(float t) {		// a szempoz�ci�t egy k�rp�ly�n mozgatjuk
		float r = sqrtf(eye.x * eye.x + eye.y * eye.y);
		eye = vec3(r * cos(t) + lookat.x, r * sin(t) + lookat.y, eye.z);
		set();
	}

	void Step(float step) {		// f�lfel� v lefel� l�p�nk egy kicsit
		eye = normalize(eye + pvup * step) * length(eye);
		set();
	}

	Ray getRay(int X, int Y) {		//x, y koordin�t�kkal megadott pixelbe mutat� sug�r a szemb�l (ray)
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + rvup * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

//---------------------------
struct Light {
	//---------------------------
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
};

GPUProgram shader;
Camera camera;
bool animate = true;

class Scene {
	Sphere sphere;
	ImplicitSurface implicitSurface;

public:
	Scene() {
		sphere = Sphere(vec3(-0.5f, 0, 0), 0.2f);
		implicitSurface = ImplicitSurface(10.5f, 10.5f, 1.5f);
	}
	/// <summary>
	/// itt hat�rozzuk meg, hogy milyen objektumokat tartalmazzon a scene, miket akarunk elmetszeni a sug�rral
	/// </summary>
	/// <param name="ray"></param>
	/// <returns></returns>
	Hit firstIntersect(Ray ray) {
		Hit bestHit;		//a szemhez legk�zelebbi metsz�spont amit a sug�r elmetsz (objektum/sug�r metsz�spont)
		Hit tempHit;
		bestHit.t = -1;
		tempHit.t = -1;
		//tempHit = convexPolyhedron.intersectConvexPolyhedron(ray, bestHit, 0.02f, 0);
		//if (tempHit.t > 0 && (bestHit.t < 0 || tempHit.t < bestHit.t))  bestHit = tempHit;

		//tempHit = convexPolyhedron.intersectConvexPolyhedron(ray, bestHit, 1.0f, 2);
		//if (tempHit.t > 0 && (bestHit.t < 0 || tempHit.t < bestHit.t))  bestHit = tempHit;

		
		tempHit = convexPolyhedron.intersectConvexPolyhedron(ray, bestHit, 1.2f, 3);		//mat = 2 -> arany; mat = 3 -> teljes visszaver�d�s
		if (tempHit.t > 0.0f && (bestHit.t < 0.0f || tempHit.t < bestHit.t))  bestHit = tempHit;		//ha ezen metsz�spont k�zelebb van a szemhez mint az el�z� akkor v�ltsuk le

		tempHit = sphere.intersect(ray, 2);
		if (tempHit.t > 0.0f && (bestHit.t < 0.0f || tempHit.t < bestHit.t))  bestHit = tempHit;		//ha ezen metsz�spont k�zelebb van a szemhez mint az el�z� akkor v�ltsuk le

		tempHit = implicitSurface.intersect(ray, 2);
		if (tempHit.t > 0.0f && (bestHit.t < 0.0f || tempHit.t < bestHit.t))  bestHit = tempHit;

		if (dot(ray.dir, bestHit.normal) > 0.0f) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 trace(Ray ray) {
		vec3 outRadiance = vec3(0, 0, 0);
		for (int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) break;
			if (hit.mat < 2) {		//rough surface
				vec3 lightdir = normalize(lightPosition - hit.position);
				float cosTheta = dot(hit.normal, lightdir);
				if (cosTheta > 0) {
					vec3 LeIn = Le / dot(lightPosition - hit.position, lightPosition - hit.position);
					outRadiance = outRadiance + (ray.weight * LeIn * kd[hit.mat] * cosTheta);
					vec3 halfway = normalize(-ray.dir + lightdir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + (ray.weight * LeIn * ks[hit.mat] * pow(cosDelta, shininess));
				}
				ray.weight = ray.weight * ka;
				break;
			}
			//mirror reflection
			//amennyiben spkekul�ris(t�r�z�) arany a material
			if (hit.mat == 2) {
				ray.weight = ray.weight * (F0Gold + (vec3(1, 1, 1) - F0Gold) * pow(dot(-ray.dir, hit.normal), 5));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			}
			//amennyiben spkekul�ris(t�r�z�) teljes visszaver�d�st biztos�t� a material
			if (hit.mat == 3) {
				ray.weight = ray.weight * (F0PerfectReflect + (vec3(1, 1, 1) - F0PerfectReflect) * pow(dot(-ray.dir, hit.normal), 5));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			}

		}
		outRadiance = outRadiance + (ray.weight * La);
		return outRadiance;
	}

	void render(std::vector<vec4>& image) {
		long timeStart = glutGet(GLUT_ELAPSED_TIME);

		for (int Y = 0; Y < windowHeight; Y++) {	//soronk�nt v�gigmegy�nk a k�perny� �sszes pixel�n
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));	//meghat�rozzuk a kamer�b�l az adott pixelbe mne� sugarat. Majd ezt v�gigk�vetj�k (trace)								
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);		//ez lesz a k�p�nk, gyakorlatilag egy ablakm�ret� 2D t�mb melyben az egyes pixelek sz�nei vannak
																						//ezt kell majd feltext�r�znunk egy n�gyzetre ami lefedi a teljes k�perny�t
			}
		}
		printf("Render Time: %d ms\n", glutGet(GLUT_ELAPSED_TIME) - timeStart);
	}

};

Scene scene;

float Fresnel(float n, float kappa) {
	return ((n - 1) * (n - 1) + kappa * kappa) / ((n + 1) * (n + 1) + kappa * kappa);
}

class FullScreenTexturedQuad {
	unsigned int vao = 0;	// vertex array object id
	unsigned int textureId = 0; //texture id
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]); // To GPU
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(shader.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	unsigned int vao;
	glGenVertexArrays(1, &vao);	// create 1 vertex array object
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;
	glGenBuffers(1, &vbo);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

	// create program for the GPU
	shader.create(vertexSourceForTexturing, fragmentSourceForTexturing, "fragmentColor");

	const float g = 0.618f;
	const float G = 1.618f;
	std::vector<vec3> vLocal = {		//a dedoka�der cs�cspont koordin�t�i
		vec3(0,g,G), vec3(0,-g,G), vec3(0,-g,-G), vec3(0,g,-G),
		vec3(G,0,g), vec3(-G,0,g), vec3(-G,0,-g), vec3(G,0,-g),
		vec3(g,G,0), vec3(-g,G,0), vec3(-g,-G,0), vec3(g,-G,0),
		vec3(1,1,1), vec3(-1,1,1), vec3(-1,-1,1), vec3(1,-1,1),
		vec3(1,-1,-1), vec3(1,1,-1), vec3(-1,1,-1), vec3(-1,-1,-1)
	};

	std::vector<int> planesLocal = {	//mely cs�cspontok tartozna ka dedoka�der egyes s�kjaihoz
		1,2,16,
		1,13,9,
		1,14,6,
		2,15,11,
		3,4,18,
		3,17,12,
		3,20,7,
		19,10,9,
		16,12,17,
		5,8,18,
		14,10,19,
		6,7,20
	};

	convexPolyhedron = ConvexPolyhedron(vLocal, planesLocal);

	//2 f�le diff�z materialunk legyen, itt adjuk meg a tulajdons�gait
	kd[0] = vec3(0.1f, 0.2f, 0.3f);
	kd[1] = vec3(1.5f, 0.6f, 0.4f);
	ks[0] = vec3(5, 5, 5);
	ks[1] = vec3(1, 1, 1);

	//speifik�ci� szerinti arany t�r�smutat�k �s kiolt�si t�nyez�k r,g,b hull�mhosszokon
	// ez�st: 0.14/4.1, 0.16/2.3, 0.13/3.1
	float redFresnel = Fresnel(0.17, 3.1);
	float greenFresnel = Fresnel(0.35, 2.7);
	float blueFresnel = Fresnel(1.5, 1.9);
	F0Gold = vec3(redFresnel, greenFresnel, blueFresnel);
	F0PerfectReflect = vec3(1.0f, 1.0f, 1.0f);		//teljes visszaver�d�s

	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad->LoadTexture(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 't') shader.setUniform(1, "top");
	if (key == 'T') shader.setUniform(0, "top");
	if (key == 'f') camera.Step(0.1f);
	if (key == 'F') camera.Step(-0.1f);
	if (key == 'a') animate = !animate;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	if (animate) {
		camera.Animate(glutGet(GLUT_ELAPSED_TIME) / 10000.0f);
	}
	glutPostRedisplay();
}