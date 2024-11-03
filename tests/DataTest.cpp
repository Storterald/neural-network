#include <gtest/gtest.h>

#include "CUDATest.h"

TEST(DataTest, DefaultConstructorInitializesEmptyData) {
        Data data{};

        EXPECT_EQ(data.data(), nullptr);
        EXPECT_EQ(data.size(), 0);
}

TEST(DataTest, DataIsCorrectlyAllocatedOnCPU) {
        Data data(10);

        EXPECT_NE(data.data(), nullptr);
        EXPECT_EQ(data.size(), 10);
        EXPECT_NO_FATAL_FAILURE(float value = data.data()[9]);
}

TEST(DataTest, DataIsCorrectlyAllocatedOnGPU) {
        Data data(10);
        for (int i{}; i < 10; ++i)
                data.data()[i] = 1;

        EXPECT_NE(data.data(), nullptr);
        EXPECT_EQ(data.size(), 10);
        EXPECT_NO_FATAL_FAILURE(CUDATest::accessValues(data));
        EXPECT_TRUE(CUDATest::checkValues(data, 1));
}

TEST(DataTest, CopyConstructorCorrectlyCopiesData) {
        Data data1(10);
        for (int i{}; i < 10; ++i)
                data1.data()[i] = 1;

        Data data2(data1);

        EXPECT_NE(data1.data(), data2.data());
        EXPECT_EQ(data2.size(), 10);
        EXPECT_TRUE([&]() -> bool {
                bool value { true };
                for (int i{}; i < 10; ++i)
                        value &= data1.data()[i] == data2.data()[i];

                return value;
        }());
}

TEST(DataTest, CopyAssignmentInitializationCorrectlyCopiesData) {
        Data data1(10);
        for (int i{}; i < 10; ++i)
                data1.data()[i] = 1;

        Data data2 = data1;

        EXPECT_NE(data1.data(), data2.data());
        EXPECT_EQ(data1.size(), data2.size());
        EXPECT_TRUE([&]() -> bool {
                bool value { true };
                for (int i{}; i < 10; ++i)
                        value &= data1.data()[i] == data2.data()[i];

                return value;
        }());
}


TEST(DataTest, CopyAssignmentCorrectlyCopiesData) {
        Data data1(10);
        for (int i{}; i < 10; ++i)
                data1.data()[i] = 1;

        Data data2(20);

        data2 = data1;

        EXPECT_NE(data1.data(), data2.data());
        EXPECT_EQ(data1.size(), data2.size());
        EXPECT_TRUE([&]() -> bool {
                bool value { true };
                for (int i{}; i < 10; ++i)
                        value &= data1.data()[i] == data2.data()[i];

                return value;
        }());
}

TEST(DataTest, MoveConstructorCorrectlyCopiesData) {
        Data data(Data(10));

        EXPECT_NE(data.data(), nullptr);
        EXPECT_EQ(data.size(), 10);
}

TEST(DataTest, MoveAssignmentInitializationCorrectlyCopiesData) {
        Data data = Data(10);

        EXPECT_NE(data.data(), nullptr);
        EXPECT_EQ(data.size(), 10);
}

TEST(DataTest, MoveAssignmentCorrectlyCopiesData) {
        Data data(20);

        data = Data(10);

        EXPECT_NE(data.data(), nullptr);
        EXPECT_EQ(data.size(), 10);
}
