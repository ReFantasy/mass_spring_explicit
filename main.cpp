#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <string>
#include <fstream>
#include <cmath>
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp> // glm::value_ptr
#include <glm\gtc\matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale, glm::perspective
#include "Utils.h"
using namespace std;



#define numVAOs 1
#define numVBOs 1

float cameraX, cameraY, cameraZ;
GLuint renderingProgram;
GLuint vao[numVAOs];
GLuint vbo[numVBOs];

// variable allocation for display
GLuint mLoc, vLoc, projLoc, tfLoc;
int width, height;


void setupVertices(void)
{
	float vertexPositions[9] =
	{ -0.5f,  0.f, -0.5f,
   0.0f, 1.0f, -0.5f,
   0.5f, 0.0f, -0.5f};

	glGenVertexArrays(1, vao);
	glBindVertexArray(vao[0]);
	glGenBuffers(numVBOs, vbo);

	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertexPositions), vertexPositions, GL_STATIC_DRAW);
}

void init(GLFWwindow* window)
{
	renderingProgram = Utils::createShaderProgram("../vertShader.glsl", "../fragShader.glsl");

	setupVertices();
}

void SetFPS(GLFWwindow* window, double currentTime);

void display(GLFWwindow* window, double currentTime) {
	glClear(GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(renderingProgram);

	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);
	glEnableVertexAttribArray(0);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glDrawArrays(GL_LINES, 0, 2);
	SetFPS(window, currentTime);
}


int main(int argc, char *argv[])
{
//	int *num_particles = 16;
//	float *spring_stiffness = 1.0;
//	float *damping = 1.0;
	//// 申请托管内存
	//cudaMallocManaged((void**)&A, sizeof(Matrix));
	//cudaMallocManaged((void**)&B, sizeof(Matrix));
	//cudaMallocManaged((void**)&C, sizeof(Matrix));
	//int nBytes = width * height * sizeof(float);
	//cudaMallocManaged((void**)&A->elements, nBytes);
	//cudaMallocManaged((void**)&B->elements, nBytes);
	//cudaMallocManaged((void**)&C->elements, nBytes);

	//// 初始化数据
	//A->height = height;
	//A->width = width;
	//B->height = height;
	//B->width = width;
	//C->height = height;
	//C->width = width;
	//for (int i = 0; i < width * height; ++i)
	//{
	//	A->elements[i] = 1.0;
	//	B->elements[i] = 2.0;
	//}

	//// 定义kernel的执行配置
	//dim3 blockSize(32, 32);
	//dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
	//	(height + blockSize.y - 1) / blockSize.y);
	//// 执行kernel
	//matMulKernel << < gridSize, blockSize >> > (A, B, C);


	//// 同步device 保证结果能正确访问
	//cudaDeviceSynchronize();
	//// 检查执行结果
	//float maxError = 0.0;
	//for (int i = 0; i < width * height; ++i)
	//	maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
	//std::cout << "最大误差: " << maxError << std::endl;

	//return 0;






	if (!glfwInit()) { exit(EXIT_FAILURE); }
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	GLFWwindow* window = glfwCreateWindow(600, 600, "Lines", NULL, NULL);
	glfwMakeContextCurrent(window);
	if (glewInit() != GLEW_OK) { exit(EXIT_FAILURE); }
	glfwSwapInterval(1);

	//glfwSetWindowSizeCallback(window, window_size_callback);

	init(window);

	while (!glfwWindowShouldClose(window)) {
		display(window, glfwGetTime());
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