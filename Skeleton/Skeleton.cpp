//=============================================================================================
// Computer Graphics Sample Program: GPU ray casting
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;          // pos of eye

	layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
	out vec3 p;		//a kamera ablak egy pixel pontja, amire épp nézünk

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";

//------------------------------------------------

//------------------------------------------------

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	const vec3 La = vec3(0.5f, 0.6f, 0.6f);
	const vec3 Le = vec3(0.8f, 0.8f, 0.8f);
	const vec3 lightPosition = vec3(0.4f, 0.4f, 0.25f);
	const vec3 ka = vec3(0.5f, 0.5f, 0.5f);
	const float shininess = 500.0f;
	const int maxdepth = 5;
	const float epsilon = 0.01f;

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;	// smooth?
	};

	struct Ray {
		vec3 start, dir, weight;
	};

	const int objFaces = 12;

	uniform int top;
	uniform vec3 wEye;
	uniform vec3 v[20];
	uniform int planes[objFaces * 3];
	uniform vec3 kd[2], ks[2], F0;

	void getObjPlane(int i, float scale,  out vec3 p,  out vec3 normal) {
		vec3 p1 = v[planes[3 * i] - 1];
		vec3 p2 = v[planes[3 * i + 1] - 1];
		vec3 p3 = v[planes[3 * i + 2] - 1];
		normal = cross(p2 - p1, p3 - p1);
		if (dot(p1, normal) < 0) normal = -normal;
		p = p1 * scale + vec3(0, 0, 0.03f);
	}

	Hit solveQuadratic(float a, float b, float c, Ray ray, Hit hit, float zmin, float zmax, float normz) {
		float discr = b * b - 4.0f * a * c;
		if (discr >= 0) {
			float sqrt_discr = sqrt(discr);
			float t1 = (-b + sqrt_discr) / 2.0f / a;
			vec3 p = ray.start + ray.dir * t1;
			if (p.z > zmax || p.z < zmin) t1 = -1;
			float t2 = (-b - sqrt_discr) / 2.0f / a;
			p = ray.start + ray.dir * t2;
			if (p.z > zmax || p.z < zmin) t2 = -1;
			if (t2 > 0 && (t2 < t1 || t1 < 0)) t1 = t2;
			if (t1 > 0 && (t1 < hit.t || hit.t < 0)) {
				hit.t = t1;
				hit.position = ray.start + ray.dir * hit.t;
				hit.normal = normalize(vec3(-hit.position.x, -hit.position.y, normz));
				hit.mat = 2;
			}
		}
		return hit;
	}

	Hit intersectMirascope(Ray ray, Hit hit) {
		const float f = 0.25f;
		const float H = 0.98f * f;

		float a = dot(ray.dir.xy, ray.dir.xy);
		float b = dot(ray.dir.xy, ray.start.xy) * 2 - 4 * f * ray.dir.z;
		float c = dot(ray.start.xy, ray.start.xy) - 4 * f * ray.start.z;
		hit = solveQuadratic(a, b, c, ray, hit, 0, f / 2, 2 * f);
		if (top == 0) return hit;
		b += 8 * f * ray.dir.z;
		c += 8 * f * ray.start.z - 4 * f * f;
		hit = solveQuadratic(a, b, c, ray, hit, f / 2, H, -2 * f);
		return hit;
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
			for (int j = 0; j < objFaces; j++) {
				if (i == j) continue;
				vec3 p11, n;
				getObjPlane(j, scale, p11, n);
				if (dot(n, intersect - p11) > 0) {
					outside = true;
					break;
				}
			}
			if (!outside) {
				hit.t = ti;
				hit.position = intersect;
				hit.normal = normalize(normal);
				hit.mat = mat;
			}
		}
		return hit;
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		bestHit = intersectMirascope(ray, bestHit);
		bestHit = intersectConvexPolyhedron(ray, bestHit, 0.02f, 0);
		bestHit = intersectConvexPolyhedron(ray, bestHit, 1.2f, 1);

		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) {
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}



	vec3 trace(Ray ray) {
		//vec3 weight = vec3(1, 1, 1);
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
			ray.weight = ray.weight * (F0 + (vec3(1, 1, 1) - F0) * pow(dot(-ray.dir, hit.normal), 5));
			ray.start = hit.position + hit.normal * epsilon;
			ray.dir = reflect(ray.dir, hit.normal);
		}
		outRadiance = outRadiance + (ray.weight * La);
		return outRadiance;
	}

	in  vec3 p;
	out vec4 fragmentColor;


	void main() {
		Ray ray;
		ray.start = wEye;
		ray.dir = normalize(p - wEye);
		ray.weight = vec3(1, 1, 1);
		fragmentColor = vec4(trace(ray), 1);
	}
)";

//---------------------------
struct Camera {
	//---------------------------
	vec3 eye, lookat, right;
	vec3 pvup;	//preferált függöleges irány
	vec3 rvup;	//valódi függöleges irány
	float fov = 45 * (float)M_PI / 180;

	Camera() : eye(0, 1, 1), pvup(0, 0, 1), lookat(0, 0, 0) {
		set();
	}
	void set() {
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(pvup, w)) * f * tanf(fov / 2);
		rvup = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	void Animate(float t) {		// a szempozíciót egy körpályán mozgatjuk
		float r = sqrtf(eye.x * eye.x + eye.y * eye.y);
		eye = vec3(r * cos(t) + lookat.x, r * sin(t) + lookat.y, eye.z);
		set();
	}

	void Step(float step) {		// fölfelé v lefelé lépünk egy kicsit
		eye = normalize(eye + pvup * step) * length(eye);
		set();
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

float Fresnel(float n, float kappa) {
	return ((n - 1) * (n - 1) + kappa * kappa) / ((n + 1) * (n + 1) + kappa * kappa);
}

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
	shader.create(vertexSource, fragmentSource, "fragmentColor");
	shader.setUniform(1, "top");

	const float g = 0.618f;
	const float G = 1.618f;
	std::vector<vec3> v = {
		vec3(0,g,G), vec3(0,-g,G), vec3(0,-g,-G), vec3(0,g,-G),
		vec3(G,0,g), vec3(-G,0,g), vec3(-G,0,-g), vec3(G,0,-g),
		vec3(g,G,0), vec3(-g,G,0), vec3(-g,-G,0), vec3(g,-G,0),
		vec3(1,1,1), vec3(-1,1,1), vec3(-1,-1,1), vec3(1,-1,1),
		vec3(1,-1,-1), vec3(1,1,-1), vec3(-1,1,-1), vec3(-1,-1,-1)
	};
	for (int i = 0; i < v.size(); i++) {
		shader.setUniform(v[i], "v[" + std::to_string(i) + "]");
	}

	std::vector<int> planes = {
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
	for (int i = 0; i < planes.size(); i++) {
		shader.setUniform(planes[i], "planes[" + std::to_string(i) + "]");
	}

	shader.setUniform(vec3(0.1f, 0.2f, 0.3f), "kd[0]");
	shader.setUniform(vec3(1.5f, 0.6f, 0.4f), "kd[1]");
	shader.setUniform(vec3(5, 5, 5), "ks[0]");
	shader.setUniform(vec3(1, 1, 1), "ks[1]");

	//speifikáció szerinti arany törésmutatók és kioltási tényezök r,g,b hullámhosszokon
	float redFresnel = Fresnel(0.17, 3.1);
	float greenFresnel = Fresnel(0.35, 2.7);
	float blueFresnel = Fresnel(1.5, 1.9);
	shader.setUniform(vec3(redFresnel, greenFresnel, blueFresnel), "F0");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	shader.setUniform(camera.eye, "wEye");
	shader.setUniform(camera.lookat, "wLookAt");
	shader.setUniform(camera.right, "wRight");
	shader.setUniform(camera.rvup, "wUp");
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
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
		camera.Animate(glutGet(GLUT_ELAPSED_TIME) / 1000.0f);
	}
	glutPostRedisplay();
}