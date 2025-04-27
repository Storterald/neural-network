#include <gtest/gtest.h>

#include <neural-network/types/Data.h>

TEST(DataTest, DefaultConstructorInitializesEmptyData) {
        Data data{};

        EXPECT_EQ(data.data(), nullptr);
        EXPECT_EQ(data.size(), 0);
}

TEST(DataTest, DataIsCorrectlyAllocatedOnCPU) {
        Data data(10);

        EXPECT_NE(data.data(), nullptr);
        EXPECT_EQ(data.size(), 10);

        Span s = data.as_span();
        EXPECT_NO_FATAL_FAILURE(float value = s[9]);
}

TEST(DataTest, CopyConstructorCorrectlyCopiesData) {
        Data data1(10);
        Span s1 = data1.as_span();
        for (uint32_t i = 0; i < 10; ++i)
                s1[i] = 1;
        s1.update();

        Data data2(data1);

        EXPECT_NE(data1.data(), data2.data());
        EXPECT_EQ(data2.size(), 10);

        Span s2 = data2.as_span();
        for (uint32_t i = 0; i < 10; ++i)
                EXPECT_EQ(s1[i], s2[i]);
}

TEST(DataTest, CopyAssignmentInitializationCorrectlyCopiesData) {
        Data data1(10);
        Span s1 = data1.as_span();
        for (uint32_t i = 0; i < 10; ++i)
                s1[i] = 1;
        s1.update();

        Data data2 = data1;

        EXPECT_NE(data1.data(), data2.data());
        EXPECT_EQ(data1.size(), data2.size());

        Span s2 = data2.as_span();
        for (uint32_t i = 0; i < 10; ++i)
                EXPECT_EQ(s1[i], s2[i]);
}


TEST(DataTest, CopyAssignmentCorrectlyCopiesData) {
        Data data1(10);
        Span s1 = data1.as_span();
        for (uint32_t i = 0; i < 10; ++i)
                s1[i] = 1;
        s1.update();

        Data data2(20);

        data2 = data1;

        EXPECT_NE(data1.data(), data2.data());
        EXPECT_EQ(data1.size(), data2.size());

        Span s2 = data2.as_span();
        for (uint32_t i = 0; i < 10; ++i)
                EXPECT_EQ(s1[i], s2[i]);
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
