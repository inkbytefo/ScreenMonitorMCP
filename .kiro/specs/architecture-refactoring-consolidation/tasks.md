# Implementation Plan

- [x] 1. AI Services Consolidation - Phase 1





  - Consolidate all AI-related functionality into a single unified service
  - Move specialized AI methods from ai_analyzer.py to ai_service.py
  - Update streaming module to use unified AI service
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 1.1 Extend core/ai_service.py with specialized AI methods


  - Add detect_ui_elements method from ai_analyzer.py to AIService class
  - Add assess_system_performance method from ai_analyzer.py to AIService class
  - Add detect_anomalies method from ai_analyzer.py to AIService class
  - Add generate_monitoring_report method from ai_analyzer.py to AIService class
  - Add extract_text method from ai_analyzer.py to AIService class
  - Add analyze_screen_for_task method from ai_analyzer.py to AIService class
  - Ensure all methods use existing analyze_image or chat_completion methods internally
  - _Requirements: 1.1, 1.2_

- [x] 1.2 Move stream_analysis_generator from ai_vision.py to streaming.py


  - Copy stream_analysis_generator function from core/ai_vision.py to core/streaming.py
  - Update function to use unified ai_service instead of AIVisionAnalyzer
  - Update import statements in streaming.py to use ai_service
  - Test that streaming analysis functionality works with unified service
  - _Requirements: 1.4_

- [x] 1.3 Update all AI service imports across the project


  - Replace imports of ai_analyzer and ai_vision with ai_service imports in mcp_server.py
  - Replace imports of ai_analyzer and ai_vision with ai_service imports in server/routes.py
  - Update any other files that import ai_analyzer or ai_vision modules
  - Ensure all AI functionality uses core.ai_service.ai_service instance
  - _Requirements: 1.5_

- [x] 1.4 Remove obsolete AI modules


  - Delete core/ai_analyzer.py file completely
  - Delete core/ai_vision.py file completely
  - Verify no remaining references to deleted modules exist
  - _Requirements: 1.2_

- [-] 2. Screen Capture Centralization - Phase 2



  - Centralize all screen capture functionality to use mss library only
  - Remove PIL.ImageGrab usage from command_handler.py
  - Update command_handler to use unified screen_capture service
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2.1 Extend core/screen_capture.py with high-quality and preview capture methods


  - Add capture_hq_frame method to ScreenCapture class for PNG high-quality captures
  - Add capture_preview_frame method to ScreenCapture class for low-quality JPEG captures
  - Implement quality and resolution parameters in new methods
  - Ensure methods return consistent data structure with existing capture_screen method
  - _Requirements: 2.1, 2.2_

- [x] 2.2 Refactor core/command_handler.py to use unified screen capture


  - Remove _capture_hq_frame_sync method that uses PIL.ImageGrab
  - Remove _capture_preview_frame_sync method that uses PIL.ImageGrab
  - Add ScreenCapture instance to CommandHandler.__init__
  - Update _handle_request_hq_frame to use self.screen_capture.capture_hq_frame
  - Update _stream_preview_frames to use self.screen_capture.capture_preview_frame
  - Maintain ThreadPoolExecutor usage for async execution
  - _Requirements: 2.2, 2.3, 2.4_

- [x] 2.3 Update core/streaming.py to use unified screen capture







  - Verify ScreenStreamer class uses core/screen_capture.py for all screen capture operations
  - Remove any direct mss usage in streaming.py if it bypasses ScreenCapture class
  - Ensure consistent screen capture behavior across all streaming operations
  - _Requirements: 2.5_

- [x] 3. Configuration Unification - Phase 3





  - Consolidate all configuration management into server/config.py
  - Remove core/config.py and update all imports
  - Ensure Pydantic BaseSettings functionality is preserved
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 3.1 Extend server/config.py with settings from core/config.py


  - Add mcp_server_name field to ServerConfig class
  - Add mcp_server_version field to ServerConfig class  
  - Add mcp_protocol_version field to ServerConfig class
  - Add default_image_format field to ServerConfig class
  - Add default_image_quality field to ServerConfig class
  - Ensure all new fields have appropriate default values and descriptions
  - _Requirements: 3.1, 3.2_

- [x] 3.2 Update all configuration imports in core modules


  - Update core/ai_service.py to import from ..server.config instead of .config
  - Update core/mcp_server.py to import from ..server.config instead of .config
  - Update any other core modules that import from core.config
  - Use relative imports (from ..server.config import config) for proper module resolution
  - _Requirements: 3.3, 3.5_

- [x] 3.3 Remove obsolete configuration module


  - Delete core/config.py file completely
  - Verify no remaining references to core.config exist in the codebase
  - Test that all configuration access works through server.config
  - _Requirements: 3.2_

- [x] 4. Protocol Layer Optimization - Phase 4





  - Minimize business logic in protocol layers (MCP and API)
  - Ensure both protocols call the same core services for identical functionality
  - Remove code duplication between mcp_server.py and server/routes.py
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 4.1 Refactor mcp_server.py to use unified services only


  - Update analyze_screen tool to use only ai_service.analyze_image
  - Update detect_ui_elements tool to use ai_service.detect_ui_elements
  - Update assess_system_performance tool to use ai_service.assess_system_performance
  - Update detect_anomalies tool to use ai_service.detect_anomalies
  - Update generate_monitoring_report tool to use ai_service.generate_monitoring_report
  - Remove any direct imports of ai_analyzer or ai_vision
  - Ensure all tools are thin wrappers that delegate to core services
  - _Requirements: 4.1, 4.2_


- [x] 4.2 Refactor server/routes.py to use unified services only

  - Update /analyze/screen endpoint to use ai_service.analyze_image
  - Remove any direct usage of AIVisionAnalyzer class
  - Ensure API endpoints delegate to the same core services as MCP tools
  - Verify that analyze_screen MCP tool and /analyze/screen API endpoint call identical core functionality
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 4.3 Remove business logic from protocol layers


  - Move any remaining business logic from mcp_server.py to appropriate core modules
  - Move any remaining business logic from server/routes.py to appropriate core modules
  - Ensure protocol layers only handle request parsing and response formatting
  - _Requirements: 4.3, 4.4_

- [x] 5. Cleanup and Validation - Phase 5




  - Remove unused imports and dependencies
  - Ensure naming consistency across the project
  - Validate that well-designed modules integrate properly with unified services
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 5.1 Clean up unused imports and dependencies


  - Search for and remove any remaining imports of deleted modules (ai_analyzer, ai_vision, core.config)
  - Remove unused import statements throughout the codebase
  - Clean up any unused variables or functions that were part of the old architecture
  - _Requirements: 5.1, 5.2_

- [x] 5.2 Ensure naming consistency across the project


  - Review function and variable names for consistency across all modules
  - Standardize naming conventions between unified services
  - Update any inconsistent naming patterns found during refactoring
  - _Requirements: 5.3_

- [x] 5.3 Validate integration of well-designed modules


  - Verify core/database_pool.py integrates correctly with unified configuration
  - Verify core/performance_monitor.py integrates correctly with unified configuration
  - Verify core/memory_system.py works properly with unified AI service
  - Test that all existing functionality is preserved after refactoring
  - _Requirements: 5.4_

- [x] 5.4 Run comprehensive testing


  - Execute all existing tests to ensure no functionality is broken
  - Test MCP tools functionality with unified services
  - Test API endpoints functionality with unified services
  - Verify streaming functionality works with consolidated services
  - Test configuration loading and usage across all modules
  - _Requirements: 5.5_

- [ ] 6. Final Quality Validation - Phase 6




  - Verify the refactored codebase meets all quality requirements
  - Ensure no code duplication exists
  - Confirm centralized service management
  - Validate improved maintainability and performance
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 6.1 Verify elimination of code duplication


  - Scan codebase for any remaining duplicate functionality
  - Ensure AI operations are handled only by unified ai_service
  - Ensure screen capture operations are handled only by unified screen_capture
  - Ensure configuration is managed only by unified server/config.py
  - _Requirements: 6.1_

- [x] 6.2 Confirm centralized service management


  - Verify AI, screen capture, and configuration services are centrally managed
  - Test that protocol layers properly delegate to core services
  - Ensure no business logic remains in protocol layers
  - _Requirements: 6.2_

- [x] 6.3 Validate improved maintainability


  - Test adding a new AI analysis method to verify ease of extension
  - Test modifying screen capture behavior to verify centralized control
  - Verify configuration changes propagate correctly throughout the system
  - _Requirements: 6.3_

- [x] 6.4 Validate improved performance


  - Measure memory usage reduction from eliminated duplicate objects
  - Verify consistent library usage (mss only for screen capture)
  - Test response times for AI operations to ensure no performance regression
  - _Requirements: 6.4_

- [x] 6.5 Confirm Single Responsibility Principle compliance


  - Review each module <to ensure it has a single, well-defined responsibility
  - Verify clear separation of concerns between protocol, service, and support layers
  - Ensure each class and function has a single reason to change
  - _Requirements: 6.5_