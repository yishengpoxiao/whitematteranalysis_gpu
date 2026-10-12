foreach(module WMAPipeline WMARegisterToAtlas WMAGroupRegister WMAAtlasBuilder WMAMergeMRML)
  slicerMacroBuildScriptedModule(NAME ${module} SCRIPTS ${module}.py)
endforeach()
install(DIRECTORY _vendor DESTINATION ${Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR}
  PATTERN "__pycache__" EXCLUDE PATTERN "*.pyc" EXCLUDE)
# Bundle the current repository backend rather than a separately installed WMA.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../whitematteranalysis
  DESTINATION ${Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR}/_vendor
  PATTERN "__pycache__" EXCLUDE PATTERN "*.pyc" EXCLUDE)
