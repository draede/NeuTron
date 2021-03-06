
#include "CX/Log/Logger.hpp"
#include "CX/Log/SystemOutput.hpp"
#include "TestSimple.hpp"
#include "TestMNIST.hpp"


using namespace CX;


int main(int argc, char *argv[])
{
	Log::Logger::GetDefaultLogger().RemoveOutputs();
	Log::Logger::GetDefaultLogger().SetLevel(Log::Level_Info);
	Log::Logger::GetDefaultLogger().AddOutput(new (std::nothrow) Log::SystemOutput());

	//return TestSimple();
	return TestMNIST();
}
