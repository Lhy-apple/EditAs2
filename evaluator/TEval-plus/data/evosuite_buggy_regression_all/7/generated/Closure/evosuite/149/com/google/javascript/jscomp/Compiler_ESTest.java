/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:23:39 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.CodeChangeHandler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerInput;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.CssRenamingMap;
import com.google.javascript.jscomp.DefaultPassConfig;
import com.google.javascript.jscomp.ErrorManager;
import com.google.javascript.jscomp.FunctionInformationMap;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.Region;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.SourceFile;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.jscomp.Tracer;
import com.google.javascript.jscomp.VariableMap;
import com.google.javascript.rhino.Node;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;
import java.util.Vector;
import java.util.logging.Logger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Compiler_ESTest extends Compiler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Logger logger0 = Logger.getLogger("Root module has no inputs");
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      Node node0 = compiler0.parseSyntheticCode("Root module has no inputs", "[|.\"-Qo)");
      compiler0.toSource(compiler_CodeBuilder0, 10, node0);
      assertEquals(2, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Logger logger0 = Logger.getLogger("Root module has no inputs");
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      // Undeclared exception!
      try { 
        compiler0.toSourceArray();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      // Undeclared exception!
      try { 
        compiler0.toSource();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Logger logger0 = Logger.getLogger("Root module has no inputs");
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      JSModule jSModule0 = new JSModule("[|.\"-Qo)");
      // Undeclared exception!
      try { 
        compiler0.toSourceArray(jSModule0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      // Undeclared exception!
      try { 
        compiler0.normalize();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      Vector<JSSourceFile> vector0 = new Vector<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) vector0, (List<JSSourceFile>) vector0, compilerOptions0);
      // Undeclared exception!
      try { 
        compiler0.check();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // One-time passes cannot be run multiple times: checkVars
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.setNormalized();
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager((Logger) null);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      compiler0.resetUniqueNameId();
      assertEquals(0, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      compiler0.parseTestCode("p^3cvbH");
      Vector<JSSourceFile> vector0 = new Vector<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) vector0, (List<JSSourceFile>) vector0, compilerOptions0);
      assertTrue(compiler0.hasErrors());
      assertEquals(1, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.toSource((Node) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Cannot build without root node being specified
         //
         verifyException("com.google.javascript.jscomp.CodePrinter$Builder", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.disableThreads();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Compiler.IntermediateState compiler_IntermediateState0 = compiler0.getState();
      assertNotNull(compiler_IntermediateState0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("com.google.javascript.jscomp.Compiler$IntermediateState");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      boolean boolean0 = compiler0.precheck();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.computeCFG();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.setUnnormalized();
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager((Logger) null);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      // Undeclared exception!
      try { 
        compiler0.setCssRenamingMap((CssRenamingMap) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Logger logger0 = Tracer.logger;
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      VariableMap variableMap0 = compiler0.getVariableMap();
      assertNull(variableMap0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      FunctionInformationMap functionInformationMap0 = compiler0.getFunctionalInformationMap();
      assertNull(functionInformationMap0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      compiler0.parseTestCode("p^3cvbH");
      // Undeclared exception!
      try { 
        compiler0.recordFunctionInformation();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      MockFile mockFile0 = new MockFile("normalize");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      compiler0.removeChangeHandler((CodeChangeHandler) null);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.setHasRegExpGlobalReferences(true);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.getMessages();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.removeTryCatchFinally();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Scope scope0 = compiler0.getTopScope();
      assertNull(scope0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getLength();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getColumnIndex();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      compiler_CodeBuilder0.reset();
      assertEquals("", compiler_CodeBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("uw8j");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(sourceFile_Generator0).getCode();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromGenerator("}|C@%Q/B", sourceFile_Generator0);
      // Undeclared exception!
      try { 
        compiler0.parse(jSSourceFile0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // FAILED ASSERTION
         //
         verifyException("com.google.javascript.jscomp.mozilla.rhino.Kit", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ArrayList<JSSourceFile> arrayList0 = new ArrayList<JSSourceFile>();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile(".");
      arrayList0.add(jSSourceFile0);
      // Undeclared exception!
      try { 
        compiler0.compile((List<JSSourceFile>) arrayList0, (List<JSSourceFile>) arrayList0, (CompilerOptions) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("", "com.google.javascript.jscomp.mozilla.rhino.JavaAdapter");
      compiler0.parse(jSSourceFile0);
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      Vector<JSModule> vector0 = new Vector<JSModule>();
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      compiler0.compileModules(stack0, vector0, compilerOptions0);
      assertEquals(1, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Logger logger0 = Logger.getLogger("Root module has no inputs");
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      compiler0.initCompilerOptionsIfTesting();
      JSModule jSModule0 = new JSModule("[|.\"-Qo)");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromGenerator("com.google.javascript.rhino.jstype.ProxyObjectType", (SourceFile.Generator) null);
      JSModule[] jSModuleArray0 = new JSModule[1];
      jSModuleArray0[0] = jSModule0;
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      // Undeclared exception!
      try { 
        compiler0.compile(jSSourceFile0, jSModuleArray0, compilerOptions0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NullPointerException
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Logger logger0 = Logger.getLogger("Root module has no inputs");
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      CompilerOptions compilerOptions0 = compiler0.options;
      DefaultPassConfig defaultPassConfig0 = new DefaultPassConfig(compilerOptions0);
      compiler0.setPassConfig(defaultPassConfig0);
      assertEquals(0, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager((Logger) null);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      // Undeclared exception!
      try { 
        compiler0.endPass();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Tracer should not be null at the end of a pass.
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      Vector<JSSourceFile> vector0 = new Vector<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile((List<JSSourceFile>) vector0, (List<JSSourceFile>) vector0, compilerOptions0);
      CompilerInput compilerInput0 = compiler0.newExternInput("// Input %num%");
      assertEquals("// Input %num%", compilerInput0.getName());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      // Undeclared exception!
      try { 
        tightenTypes0.getTypeRegistry();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.getReverseAbstractInterpreter();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      compiler0.parseTestCode("p^3cvbH");
      // Undeclared exception!
      try { 
        compiler0.parseInputs();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      boolean boolean0 = compiler_CodeBuilder0.endsWith(":\"%F:WeIly:W{Xq\"bZp");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      Compiler.CodeBuilder compiler_CodeBuilder1 = compiler_CodeBuilder0.append("// Input %num%");
      boolean boolean0 = compiler_CodeBuilder1.endsWith("[|.\"-Qo)");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Logger logger0 = Logger.getLogger("Root module has no inputs");
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      compiler0.reportCodeChange();
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      compiler0.parseTestCode("p^3cvbH");
      compiler0.parseSyntheticCode("p^3cvbH");
      assertEquals(2, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      compiler0.parseTestCode("p^3cvbH");
      // Undeclared exception!
      try { 
        compiler0.check();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Region region0 = compiler0.getSourceRegion("Va$x6V1bu|*#", 0);
      assertNull(region0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      String string0 = compiler0.getAstDotGraph();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ErrorManager errorManager0 = compiler0.getErrorManager();
      assertEquals(0, errorManager0.getWarningCount());
  }
}