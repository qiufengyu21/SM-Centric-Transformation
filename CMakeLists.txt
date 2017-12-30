set(LLVM_LINK_COMPONENTS support)

add_clang_executable(smc
	smc.cpp
    )
    target_link_libraries(smc
      clangTooling
        clangBasic
	  clangASTMatchers
	    )
