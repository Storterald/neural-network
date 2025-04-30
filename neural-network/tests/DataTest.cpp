#include <gtest/gtest.h>

#include <neural-network/types/buf.h>

TEST(DataTest, DefaultConstructorInitializesEmptyData) {
        const nn::buf buffer{};

        EXPECT_EQ(buffer.data(), nullptr);
        EXPECT_EQ(buffer.size(), 0);
}

TEST(DataTest, DataIsCorrectlyAllocatedOnCPU) {
        const nn::buf buffer(10);

        EXPECT_NE(buffer.data(), nullptr);
        EXPECT_EQ(buffer.size(), 10);

        const nn::span s = buffer.as_span();
        EXPECT_NO_FATAL_FAILURE(float value = s[9]);
}

TEST(DataTest, CopyConstructorCorrectlyCopiesData) {
        nn::buf buffer1(10);
        nn::span s1 = buffer1.as_span();
        for (uint32_t i = 0; i < 10; ++i)
                s1[i] = 1;
        s1.update();

        const nn::buf buffer2(buffer1);

        EXPECT_NE(buffer1.data(), buffer2.data());
        EXPECT_EQ(buffer2.size(), 10);

        const nn::span s2 = buffer2.as_span();
        for (uint32_t i = 0; i < 10; ++i)
                EXPECT_EQ(s1[i], s2[i]);
}

TEST(DataTest, CopyAssignmentCorrectlyCopiesData) {
        nn::buf buffer1(10);
        nn::span s1 = buffer1.as_span();
        for (uint32_t i = 0; i < 10; ++i)
                s1[i] = 1;
        s1.update();

        nn::buf buffer2(20);

        buffer2 = buffer1;

        EXPECT_NE(buffer1.data(), buffer2.data());
        EXPECT_EQ(buffer1.size(), buffer2.size());

        nn::span s2 = buffer2.as_span();
        for (uint32_t i = 0; i < 10; ++i)
                EXPECT_EQ(s1[i], s2[i]);
}

TEST(DataTest, MoveConstructorCorrectlyCopiesData) {
        const nn::buf buffer(nn::buf(10));

        EXPECT_NE(buffer.data(), nullptr);
        EXPECT_EQ(buffer.size(), 10);
}

TEST(DataTest, MoveAssignmentCorrectlyCopiesData) {
        nn::buf buffer(20);

        buffer = nn::buf(10);

        EXPECT_NE(buffer.data(), nullptr);
        EXPECT_EQ(buffer.size(), 10);
}
