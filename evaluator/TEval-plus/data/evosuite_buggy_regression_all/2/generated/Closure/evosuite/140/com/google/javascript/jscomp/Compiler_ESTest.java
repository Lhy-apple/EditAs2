/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:38:57 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Supplier;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.CssRenamingMap;
import com.google.javascript.jscomp.DefaultPassConfig;
import com.google.javascript.jscomp.ErrorManager;
import com.google.javascript.jscomp.FunctionInformationMap;
import com.google.javascript.jscomp.GatherRawExports;
import com.google.javascript.jscomp.GoogleCodingConvention;
import com.google.javascript.jscomp.JSError;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.JsAst;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.PrintStreamErrorManager;
import com.google.javascript.jscomp.Region;
import com.google.javascript.jscomp.ReverseAbstractInterpreter;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.SourceFile;
import com.google.javascript.jscomp.SourceMap;
import com.google.javascript.jscomp.SymbolTable;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.jscomp.VariableMap;
import com.google.javascript.rhino.Node;
import java.io.File;
import java.nio.charset.Charset;
import java.util.Set;
import java.util.logging.Level;
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
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("v");
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      compiler0.toSource(compiler_CodeBuilder0, 49, node0);
      assertEquals("v;", compiler_CodeBuilder0.toString());
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.toSource((JSModule) null);
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
      Compiler compiler0 = new Compiler();
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
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
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
  public void test04()  throws Throwable  {
      File file0 = MockFile.createTempFile("6cW#(!UH29'*aL", "6cW#(!UH29'*aL");
      MockPrintStream mockPrintStream0 = new MockPrintStream(file0);
      PrintStreamErrorManager printStreamErrorManager0 = new PrintStreamErrorManager(mockPrintStream0);
      Compiler compiler0 = new Compiler(printStreamErrorManager0);
      JSModule jSModule0 = new JSModule("com.google.common.collect.LinkedListMultimap$MultisetView");
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
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
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
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.setNormalized();
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("p.Compiler$4");
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      JSModule[] jSModuleArray0 = new JSModule[0];
      // Undeclared exception!
      try { 
        compiler0.compile(jSSourceFile0, jSModuleArray0, (CompilerOptions) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.resetUniqueNameId();
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Level level0 = Level.CONFIG;
      Compiler.setLoggingLevel(level0);
      assertEquals("CONFIG", level0.toString());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.getCssRenamingMap();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseSyntheticCode("X/<MZL8?K#s;w>", "");
      compiler0.getErrorManager();
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Compiler.IntermediateState compiler_IntermediateState0 = compiler0.getState();
      compiler0.setState(compiler_IntermediateState0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      assertNull(compilerOptions0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.disableThreads();
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ScopeCreator scopeCreator0 = compiler0.getScopeCreator();
      assertNull(scopeCreator0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.rebuildInputsFromModules();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      boolean boolean0 = compiler0.precheck();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
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
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      boolean boolean0 = compiler0.isNormalized();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      VariableMap variableMap0 = compiler0.getPropertyMap();
      assertNull(variableMap0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscomp");
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
        compiler0.processDefines();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.DefaultPassConfig", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("toSourceArray");
      // Undeclared exception!
      try { 
        compiler0.optimize();
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
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Compiler compiler0 = new Compiler();
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
  public void test26()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.getWarningCount();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getVariableMap();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      DefaultPassConfig defaultPassConfig0 = new DefaultPassConfig(compilerOptions0);
      // Undeclared exception!
      try { 
        compiler0.setPassConfig(defaultPassConfig0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // this.passes has already been assigned
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      FunctionInformationMap functionInformationMap0 = compiler0.getFunctionalInformationMap();
      assertNull(functionInformationMap0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      assertNotNull(supplier0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.recordFunctionInformation();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
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
  public void test32()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(sourceFile_Generator0).getCode();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromGenerator("d9G]7Vzr/OtAt!9;", sourceFile_Generator0);
      Logger logger0 = Logger.getAnonymousLogger((String) null);
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      JSSourceFile jSSourceFile1 = JSSourceFile.fromCode("E.", "runCustomPasses");
      // Undeclared exception!
      try { 
        compiler0.compile(jSSourceFile1, jSSourceFile0, compilerOptions0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: FAILED ASSERTION
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSModule jSModule0 = new JSModule("// Input %num%");
      // Undeclared exception!
      try { 
        compiler0.getNodeForCodeInsertion(jSModule0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SourceMap sourceMap0 = compiler0.getSourceMap();
      assertNull(sourceMap0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.isTypeCheckingEnabled();
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
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.getRoot();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SymbolTable symbolTable0 = compiler0.acquireSymbolTable();
      compiler0.removeChangeHandler(symbolTable0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("q7g7q:&]");
      GatherRawExports gatherRawExports0 = new GatherRawExports(compiler0);
      Set<String> set0 = gatherRawExports0.getExportedVariableNames();
      // Undeclared exception!
      try { 
        compiler0.stripCode(set0, set0, set0, set0);
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
  public void test39()  throws Throwable  {
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      JSError[] jSErrorArray0 = compiler0.getMessages();
      assertEquals(0, jSErrorArray0.length);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
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
  public void test41()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Scope scope0 = compiler0.getTopScope();
      assertNull(scope0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getLength();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      String string0 = compiler_CodeBuilder0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getLineIndex();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      compiler_CodeBuilder0.reset();
      assertEquals("", compiler_CodeBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Logger logger0 = Logger.getLogger("lxIInP?|%$Udnx");
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      compiler0.acquireSymbolTable();
      // Undeclared exception!
      try { 
        compiler0.acquireSymbolTable();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // SymbolTable already acquired
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      MockFile mockFile0 = new MockFile("amnz$X;j98NAf=");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      compiler0.parseTestCode("amnz$X;j98NAf=");
      assertEquals(1, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("amnz$X;j98NAf=");
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[6];
      Charset charset0 = Charset.defaultCharset();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("JSC_DUPLICATE_INPUT_IN_MODULES_ERROR", charset0);
      jSSourceFileArray0[0] = jSSourceFile0;
      jSSourceFileArray0[1] = jSSourceFileArray0[0];
      jSSourceFileArray0[2] = jSSourceFile0;
      jSSourceFileArray0[3] = jSSourceFile0;
      jSSourceFileArray0[4] = jSSourceFile0;
      jSSourceFileArray0[5] = jSSourceFileArray0[3];
      CompilerOptions compilerOptions0 = compiler0.options;
      compiler0.compile(jSSourceFileArray0, jSSourceFileArray0, compilerOptions0);
      assertEquals(3, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      doReturn("9G]7Vzr/Ott!9;").when(sourceFile_Generator0).getCode();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromGenerator("9G]7Vzr/Ott!9;", sourceFile_Generator0);
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      compiler0.parse();
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[2];
      // Undeclared exception!
      try { 
        compiler0.compile(jSSourceFileArray0, jSSourceFileArray0, compilerOptions0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DefaultPassConfig defaultPassConfig0 = new DefaultPassConfig((CompilerOptions) null);
      compiler0.setPassConfig(defaultPassConfig0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("amnz$X;j98NAf=");
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
  public void test52()  throws Throwable  {
      Compiler compiler0 = new Compiler();
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
  public void test53()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.reportCodeChange();
      compiler0.parseTestCode("8");
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("VCrgFR");
      compiler0.areNodesEqualForInlining(node0, node0);
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      MockFile mockFile0 = new MockFile((String) null, "// Input %num%");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile((File) mockFile0, (Charset) null);
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      // Undeclared exception!
      try { 
        compiler0.newExternInput("h\"sA8");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      JSSourceFile jSSourceFile0 = JSSourceFile.fromGenerator("qQzUxXc^g&", sourceFile_Generator0);
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      // Undeclared exception!
      try { 
        compiler0.newExternInput("qQzUxXc^g&");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Conflicting externs name: qQzUxXc^g&
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MockFile mockFile0 = new MockFile("837e+Q=]1C", "837e+Q=]1C");
      CompilerOptions compilerOptions0 = new CompilerOptions();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile((File) mockFile0, (Charset) null);
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      JsAst jsAst0 = new JsAst(jSSourceFile0);
      // Undeclared exception!
      try { 
        compiler0.addIncrementalSourceAst(jsAst0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Duplicate input of name /data/lhy/TEval-plus/837e+Q=]1C/837e+Q=]1C
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("8");
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("8", "q|Q.HMp#(@bC^5");
      JsAst jsAst0 = new JsAst(sourceFile_Preloaded0);
      compiler0.addIncrementalSourceAst(jsAst0);
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseSyntheticCode("k&4}r(:E", "k&4}r(:E");
      ReverseAbstractInterpreter reverseAbstractInterpreter0 = compiler0.getReverseAbstractInterpreter();
      assertNotNull(reverseAbstractInterpreter0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("+z#");
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[1];
      jSSourceFileArray0[0] = jSSourceFile0;
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile(jSSourceFile0, jSSourceFileArray0, compilerOptions0);
      compiler0.parseTestCode("+z#");
      assertEquals(2, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      compiler_CodeBuilder0.append("Awc:oFFJE}RX");
      boolean boolean0 = compiler_CodeBuilder0.endsWith("h\"sA8");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      Compiler.CodeBuilder compiler_CodeBuilder1 = compiler_CodeBuilder0.append(" \r\n\t\u3000\u00A0\u2007\u202F");
      assertSame(compiler_CodeBuilder0, compiler_CodeBuilder1);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getColumnIndex();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      boolean boolean0 = compiler_CodeBuilder0.endsWith("duplicate child");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      Compiler.CodeBuilder compiler_CodeBuilder1 = compiler_CodeBuilder0.append("B6\"g@WE'Qi\"@qiZKSF");
      boolean boolean0 = compiler_CodeBuilder1.endsWith("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("+z#");
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[1];
      jSSourceFileArray0[0] = jSSourceFile0;
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile(jSSourceFile0, jSSourceFileArray0, compilerOptions0);
      compiler0.optimize();
      assertFalse(compiler0.isTypeCheckingEnabled());
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("VCrgFR");
      boolean boolean0 = compiler0.isInliningForbidden();
      assertFalse(compiler0.hasErrors());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      doReturn("9G]7Vzr/Ott!9;").when(sourceFile_Generator0).getCode();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromGenerator("9G]7Vzr/Ott!9;", sourceFile_Generator0);
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      compiler0.parse();
      compiler0.getNodeForCodeInsertion((JSModule) null);
      assertEquals(4, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("8");
      CheckLevel checkLevel0 = CheckLevel.OFF;
      String[] stringArray0 = new String[9];
      JSError jSError0 = JSError.make("8", (-1043), 20, checkLevel0, compiler0.MOTION_ITERATIONS_ERROR, stringArray0);
      compiler0.report(jSError0);
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.throwInternalError("fJ2TmnzrKcJ20H}", (Exception) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // fJ2TmnzrKcJ20H}
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("hsA8");
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(sourceFile_Generator0).getCode();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromGenerator(" [testcode] ", sourceFile_Generator0);
      CompilerOptions compilerOptions0 = compiler0.options;
      // Undeclared exception!
      try { 
        compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.SourceFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Region region0 = compiler0.getSourceRegion("-1", (-1600));
      assertNull(region0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("amnz$X;j98NAf=");
      Region region0 = compiler0.getSourceRegion("amnz$X;j98NAf=", 1);
      assertNull(region0);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSModule jSModule0 = new JSModule("8");
      Charset charset0 = Charset.defaultCharset();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("(nI]7bfz C:<;s10`", charset0);
      jSModule0.addFirst(jSSourceFile0);
      // Undeclared exception!
      try { 
        compiler0.getNodeForCodeInsertion(jSModule0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      String string0 = compiler0.getAstDotGraph();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      doReturn("9G]7Vzr/Ott!9;").when(sourceFile_Generator0).getCode();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromGenerator("9G]7Vzr/Ott!9;", sourceFile_Generator0);
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      assertEquals(1, compiler0.getErrorCount());
      
      compiler0.parse();
      compiler0.getAstDotGraph();
      assertEquals(4, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ErrorManager errorManager0 = compiler0.getErrorManager();
      assertEquals(0, errorManager0.getErrorCount());
  }
}