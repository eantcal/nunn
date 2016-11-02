find . -name \*.h -exec clang-format -i -style="{BasedOnStyle: Mozilla, IndentWidth: 4, AlwaysBreakAfterDefinitionReturnType: None, AlwaysBreakAfterReturnType: None, MaxEmptyLinesToKeep: 2}" {} \;
find . -name \*.cc -exec clang-format -i -style="{BasedOnStyle: Mozilla, IndentWidth: 4, AlwaysBreakAfterDefinitionReturnType: None, AlwaysBreakAfterReturnType: None, MaxEmptyLinesToKeep: 2}" {} \;
