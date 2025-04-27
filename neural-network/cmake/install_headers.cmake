if (EXISTS ${INCLUDE_DIR})
        file(REMOVE_RECURSE ${INCLUDE_DIR})
endif ()

file(COPY "${SRC_DIR}/" DESTINATION "${INCLUDE_DIR}/neural-network"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh")