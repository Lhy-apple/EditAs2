/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:39:47 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Supplier;
import com.google.javascript.jscomp.CheckAccessControls;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.CoalesceVariableNames;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerInput;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.DefaultPassConfig;
import com.google.javascript.jscomp.ErrorFormat;
import com.google.javascript.jscomp.ErrorManager;
import com.google.javascript.jscomp.JSError;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.LightweightMessageFormatter;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.MessageFormatter;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.PassConfig;
import com.google.javascript.jscomp.PrintStreamErrorManager;
import com.google.javascript.jscomp.ProcessClosurePrimitives;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.SourceExcerptProvider;
import com.google.javascript.jscomp.SourceMap;
import com.google.javascript.rhino.Node;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Compiler_ESTest extends Compiler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Vector<JSSourceFile> vector0 = new Vector<JSSourceFile>();
      compiler0.compile((List<JSSourceFile>) vector0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
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
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      Node node0 = Node.newString(4812, ":Mp");
      // Undeclared exception!
      try { 
        compiler0.toSource(compiler_CodeBuilder0, 4812, node0);
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
  public void test04()  throws Throwable  {
      Level level0 = Level.ALL;
      Compiler.setLoggingLevel(level0);
      assertEquals("ALL", level0.getName());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
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
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Compiler.IntermediateState compiler_IntermediateState0 = compiler0.getState();
      compiler0.setState(compiler_IntermediateState0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("");
      String[] stringArray0 = new String[6];
      JSError jSError0 = JSError.make("", 34, 0, compiler0.OPTIMIZE_LOOP_ERROR, stringArray0);
      LightweightMessageFormatter lightweightMessageFormatter0 = new LightweightMessageFormatter(compiler0);
      lightweightMessageFormatter0.formatError(jSError0);
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      assertNull(compilerOptions0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.disableThreads();
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ScopeCreator scopeCreator0 = compiler0.getScopeCreator();
      assertNull(scopeCreator0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
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
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      ProcessClosurePrimitives processClosurePrimitives0 = new ProcessClosurePrimitives(compiler0, checkLevel0, true);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, processClosurePrimitives0);
      // Undeclared exception!
      try { 
        nodeTraversal0.getModule();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CoalesceVariableNames coalesceVariableNames0 = new CoalesceVariableNames(compiler0, true);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("prefix must start with one of: ");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.getWarnings();
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
  public void test17()  throws Throwable  {
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
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parse();
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
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      assertNotNull(supplier0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SourceMap sourceMap0 = compiler0.getSourceMap();
      assertNull(sourceMap0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.getResult();
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
      Node node0 = compiler0.getRoot();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      Logger logger0 = Logger.getLogger("l@uu7h\"~mkqgK*e");
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(lightweightMessageFormatter0, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      List<CompilerInput> list0 = compiler0.getInputsForTesting();
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
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
  public void test25()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.setHasRegExpGlobalReferences(false);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(0);
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0);
      Compiler compiler0 = new Compiler(mockPrintStream0);
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
  public void test27()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Scope scope0 = compiler0.getTopScope();
      assertNull(scope0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.addToDebugLog("");
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getLength();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      String string0 = compiler_CodeBuilder0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getLineIndex();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getColumnIndex();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      compiler_CodeBuilder0.reset();
      assertEquals("", compiler_CodeBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      DefaultPassConfig defaultPassConfig0 = new DefaultPassConfig(compilerOptions0);
      PassConfig.PassConfigDelegate passConfig_PassConfigDelegate0 = new PassConfig.PassConfigDelegate(defaultPassConfig0);
      compiler0.setPassConfig(passConfig_PassConfigDelegate0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DefaultPassConfig defaultPassConfig0 = new DefaultPassConfig((CompilerOptions) null);
      // Undeclared exception!
      try { 
        defaultPassConfig0.makeTypeCheck(compiler0);
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
      CheckAccessControls checkAccessControls0 = null;
      try {
        checkAccessControls0 = new CheckAccessControls(compiler0);
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
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      compiler0.parseTestCode("the \"arguments\" object cannot be reassigned in ES5 strict mode");
      assertEquals(1, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      boolean boolean0 = compiler_CodeBuilder0.endsWith("KI?8@>%4~~'05w3:3");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      compiler_CodeBuilder0.append("Exceeded max number of optimization iterations: {0}");
      boolean boolean0 = compiler_CodeBuilder0.endsWith("'1S");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      Compiler.CodeBuilder compiler_CodeBuilder1 = compiler_CodeBuilder0.append("wJS7mWtjQ%m+lL2");
      compiler_CodeBuilder1.append("Exceeded max number of optimization iterations: {0}");
      boolean boolean0 = compiler_CodeBuilder0.endsWith("Exceeded max number of optimization iterations: {0}");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SourceExcerptProvider.SourceExcerpt sourceExcerptProvider_SourceExcerpt0 = SourceExcerptProvider.SourceExcerpt.LINE;
      LightweightMessageFormatter lightweightMessageFormatter0 = new LightweightMessageFormatter(compiler0, sourceExcerptProvider_SourceExcerpt0);
      String[] stringArray0 = new String[4];
      JSError jSError0 = JSError.make("mAoSe*2", 0, (-1353), compiler0.MOTION_ITERATIONS_ERROR, stringArray0);
      String string0 = lightweightMessageFormatter0.formatError(jSError0);
      assertEquals("mAoSe*2: ERROR - Exceeded max number of code motion iterations: null\n", string0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      String[] stringArray0 = new String[4];
      JSError jSError0 = JSError.make("7XRehJ0y_'x(T^~n", 609, (-55), checkLevel0, compiler0.MOTION_ITERATIONS_ERROR, stringArray0);
      ErrorFormat errorFormat0 = ErrorFormat.LEGACY;
      MessageFormatter messageFormatter0 = errorFormat0.toFormatter(compiler0, false);
      // Undeclared exception!
      try { 
        jSError0.format(checkLevel0, messageFormatter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      String[] stringArray0 = new String[23];
      JSError jSError0 = JSError.make("mAoSe*2", 0, (-1546), checkLevel0, compiler0.MOTION_ITERATIONS_ERROR, stringArray0);
      ErrorFormat errorFormat0 = ErrorFormat.MULTILINE;
      MessageFormatter messageFormatter0 = errorFormat0.toFormatter(compiler0, true);
      String string0 = jSError0.format(checkLevel0, messageFormatter0);
      assertEquals("mAoSe*2: \u001B[31mERROR\u001B[39m - Exceeded max number of code motion iterations: null\n", string0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      PrintStreamErrorManager printStreamErrorManager0 = new PrintStreamErrorManager((PrintStream) null);
      Compiler compiler0 = new Compiler(printStreamErrorManager0);
      JSModule jSModule0 = new JSModule("cId7.Q");
      // Undeclared exception!
      try { 
        compiler0.getNodeForCodeInsertion(jSModule0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Root module has no inputs
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      String string0 = compiler0.getAstDotGraph();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ErrorManager errorManager0 = compiler0.getErrorManager();
      assertEquals(0, errorManager0.getErrorCount());
  }
}
