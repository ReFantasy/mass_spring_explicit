#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "eigen3/Eigen/Dense"
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <string>
#include <fstream>
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp> // glm::value_ptr
#include <glm\gtc\matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale, glm::perspective
#include "Utils.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#include <random>
//using namespace std;


float RandomNumber(const float& lo = 0.0, const float& hi = 1.0)
{
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<float> dist(lo, hi);
	return dist(rng);
}



#define numVAOs 1
#define numVBOs 1

float cameraX, cameraY, cameraZ;
GLuint renderingProgram;
GLuint vao[numVAOs];
GLuint vbo[numVBOs];

// variable allocation for display
GLuint mLoc, vLoc, projLoc, tfLoc;
int width, height;

int num_particles = 3;
__device__ int n = 3;

void setupVertices(float * vertexPositions)
{
	/*float vertexPositions[9] =
	{ -0.5f,  0.f, -0.5f,
   0.0f, 1.0f, -0.5f,
   0.5f, 0.0f, -0.5f};*/


	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * num_particles * 3, vertexPositions, GL_STATIC_DRAW);
}

void init(GLFWwindow* window)
{
	renderingProgram = Utils::createShaderProgram("../vertShader.glsl", "../fragShader.glsl");

	glGenVertexArrays(1, vao);
	glBindVertexArray(vao[0]);
	glGenBuffers(numVBOs, vbo);
}

void SetFPS(GLFWwindow* window, double currentTime);

void display(GLFWwindow* window, double currentTime, float* vertexPositions)
{
	glClear(GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0, 0.2, 0.7, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(renderingProgram);


	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * num_particles * 3, vertexPositions, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);
	glEnableVertexAttribArray(0);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glDrawArrays(GL_TRIANGLES, 0, 3);

	SetFPS(window, currentTime);
}

__global__ void Kernel(float * x, float* v, float** rest_length)
{
	float dt = 1.0/(2.72*2.72*2.72);
	float spring_stiffness = 10000;
	float damping = 20;
	float particle_mass = 1000000.0;
	

	int i = threadIdx.x;
	v[i] *= 1.0 / ((-dt * damping) * (-dt * damping) * (-dt * damping)); //exp(-dt * damping);
	
	Eigen::Vector3f gravity(0,-9.8,0);
	Eigen::Vector3f total_force = gravity * particle_mass;

	Eigen::Vector3f V(0, 0, 0);
	for(int j = 0; j < n; j++)
	{
		if (rest_length[i][j] != 0)
		{
			//printf("%d->%d:%d\n", i, j, 1);
			Eigen::Vector3f x_ij(x[i * 3]- x[j * 3], x[i * 3 + 1]- x[j * 3 + 1], x[i * 3 + 2]- x[j * 3 + 2]);
			total_force += ( -spring_stiffness * (x_ij.norm() - rest_length[i][j]) )*x_ij.normalized();
		}
		V += dt * total_force / particle_mass;
	}
	v[i * 3] += V.x();
	v[i * 3+1] += V.y();
	v[i * 3+2] += V.z();

	x[i * 3] += v[i * 3] * dt;
	x[i * 3+1] += v[i * 3+1] * dt;
	x[i * 3+2] += v[i * 3+2] * dt;

}

int main(int argc, char *argv[])
{
//	int *num_particles = 16;
//	float *spring_stiffness = 1.0;
//	float *damping = 1.0;
	//exp(4);
	// 
	float* x;
	float* v;
	float** rest_length;
	cudaMallocManaged((void**)&x, sizeof(float)*num_particles*3);
	cudaMallocManaged((void**)&v, sizeof(float) * num_particles*3);
	cudaMallocManaged((void**)&rest_length, sizeof(float*) * num_particles);
	for(int i = 0;i<num_particles;i++)
	    cudaMallocManaged((void**)&rest_length[i], sizeof(float) * num_particles);
	for (int i = 0; i < num_particles; i++)
		for (int j = 0; j < num_particles; j++)
			rest_length[i][j] = 1;

	for (int i = 0; i < num_particles; i++)
		{
			x[i * 3] = RandomNumber(-0.1, 0.1);
			x[i * 3 + 1] = RandomNumber(-0.1, 0.1);
			x[i * 3 + 2] = -0.5;
			v[i*3] = 0;
			v[i * 3+1] = 0;
			v[i * 3+2] = 0;
		}

	
	if (!glfwInit()) { exit(EXIT_FAILURE); }
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	GLFWwindow* window = glfwCreateWindow(600, 600, "Lines", NULL, NULL);
	glfwMakeContextCurrent(window);
	if (glewInit() != GLEW_OK) { exit(EXIT_FAILURE); }
	glfwSwapInterval(1);

	//glfwSetWindowSizeCallback(window, window_size_callback);

	init(window);
	setupVertices(x);

	while (!glfwWindowShouldClose(window))
	{
		/*for (int i = 0; i < num_particles; i++)
		{
			vertexPositions[i * 3] = RandomNumber(-1, 1);
			vertexPositions[i * 3 + 1] = RandomNumber(-1, 1);
			vertexPositions[i * 3 + 2] = -0.5;
		}*/
		Kernel << <1, 3 >> > (x,v, rest_length);
		cudaDeviceSynchronize();

		
		cudaDeviceSynchronize();

		display(window, glfwGetTime(), x);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);



}
void SetFPS(GLFWwindow* window, double currentTime)
{
	static int n = 0;
	static double last_time = 0;
	static constexpr int interval = 10;

	if (n == interval)
	{
		int fps = static_cast<int>(interval / (currentTime - last_time));
		glfwSetWindowTitle(window, ("FPS: " + std::to_string(fps)).c_str());
		last_time = currentTime;
		n = 0;
	}
	n++;
}