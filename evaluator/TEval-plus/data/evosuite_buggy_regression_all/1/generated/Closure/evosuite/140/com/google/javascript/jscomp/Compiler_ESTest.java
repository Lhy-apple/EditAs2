/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:17:25 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Supplier;
import com.google.javascript.jscomp.CodeChangeHandler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.CssRenamingMap;
import com.google.javascript.jscomp.DefaultPassConfig;
import com.google.javascript.jscomp.ErrorManager;
import com.google.javascript.jscomp.FunctionInformationMap;
import com.google.javascript.jscomp.InlineGetters;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.JsAst;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.PassConfig;
import com.google.javascript.jscomp.Region;
import com.google.javascript.jscomp.Result;
import com.google.javascript.jscomp.ReverseAbstractInterpreter;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.SourceMap;
import com.google.javascript.jscomp.TypeValidator;
import com.google.javascript.jscomp.VariableMap;
import com.google.javascript.rhino.Node;
import com.google.protobuf.ByteString;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.SequenceInputStream;
import java.nio.charset.Charset;
import java.util.Enumeration;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Compiler_ESTest extends Compiler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("", (Charset) null);
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      compiler0.toSource();
      assertEquals(1, compiler0.getErrorCount());
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
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      // Undeclared exception!
      try { 
        compiler0.toSource(compiler_CodeBuilder0, (-1368), (Node) null);
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
      JSModule jSModule0 = new JSModule(" [testcode] ");
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
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      InlineGetters inlineGetters0 = new InlineGetters(compiler0);
      Set<String> set0 = inlineGetters0.nonMethodProperties;
      // Undeclared exception!
      try { 
        compiler0.stripCode(set0, set0, (Set<String>) null, set0);
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
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[0];
      JSModule[] jSModuleArray0 = new JSModule[0];
      compiler0.compile(jSSourceFileArray0, jSModuleArray0, compilerOptions0);
      compiler0.parseInputs();
      // Undeclared exception!
      try { 
        compiler0.compile(jSSourceFileArray0, jSModuleArray0, compilerOptions0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.resetUniqueNameId();
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Level level0 = Level.FINEST;
      Compiler.setLoggingLevel(level0);
      assertEquals(300, level0.intValue());
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
      Node node0 = compiler0.parseSyntheticCode("<li>REFERENCED BY: ", "<li>REFERENCED BY: ");
      assertEquals(7, Node.LOCAL_PROP);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Compiler.IntermediateState compiler_IntermediateState0 = compiler0.getState();
      compiler0.setState(compiler_IntermediateState0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.disableThreads();
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ScopeCreator scopeCreator0 = compiler0.getScopeCreator();
      assertNull(scopeCreator0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
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
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      boolean boolean0 = compiler0.precheck();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
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
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      boolean boolean0 = compiler0.isNormalized();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      VariableMap variableMap0 = compiler0.getPropertyMap();
      assertNull(variableMap0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("arguments");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
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
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
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
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TypeValidator typeValidator0 = compiler0.getTypeValidator();
      assertNotNull(typeValidator0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
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
  public void test25()  throws Throwable  {
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
  public void test26()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      VariableMap variableMap0 = compiler0.getVariableMap();
      assertNull(variableMap0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ByteString.Output byteString_Output0 = ByteString.newOutput();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteString_Output0);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      FunctionInformationMap functionInformationMap0 = compiler0.getFunctionalInformationMap();
      assertNull(functionInformationMap0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
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
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile(" [testcode] ", (Charset) null);
      JSSourceFile jSSourceFile1 = JSSourceFile.fromFile("", (Charset) null);
      Result result0 = compiler0.compile(jSSourceFile0, jSSourceFile1, compilerOptions0);
      assertFalse(result0.success);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SourceMap sourceMap0 = compiler0.getSourceMap();
      assertNull(sourceMap0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
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
  public void test35()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.getRoot();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CodeChangeHandler.RecentChange codeChangeHandler_RecentChange0 = compiler0.recentChange;
      compiler0.removeChangeHandler(codeChangeHandler_RecentChange0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      // Undeclared exception!
      try { 
        compiler0.removeTryCatchFinally();
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
  public void test38()  throws Throwable  {
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager((Logger) null);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      assertEquals(0, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode((String) null, (String) null);
      CompilerOptions compilerOptions0 = new CompilerOptions();
      // Undeclared exception!
      try { 
        compiler0.compile(jSSourceFile0, (JSModule[]) null, compilerOptions0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
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
      Compiler compiler0 = new Compiler();
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
      MockPrintStream mockPrintStream0 = new MockPrintStream("'ujg|$");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.initOptions((CompilerOptions) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("wnQ[CPe'lh!%\"<pQ,[3");
      JSModule jSModule0 = new JSModule("e+ +l<zatQ");
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[1];
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("GuCHY_K Q`CU&''G", "2-ZgL>.Hth");
      JSModule[] jSModuleArray0 = new JSModule[7];
      jSModuleArray0[0] = jSModule0;
      jSModule0.add(jSSourceFile0);
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      // Undeclared exception!
      try { 
        compiler0.compile(jSSourceFileArray0, jSModuleArray0, compilerOptions0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.JsAst", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSModule jSModule0 = new JSModule("*eCS,#MY[~");
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[1];
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("d;bmH_'G");
      jSSourceFileArray0[0] = jSSourceFile0;
      JSModule[] jSModuleArray0 = new JSModule[7];
      jSModuleArray0[0] = jSModule0;
      jSModuleArray0[1] = jSModule0;
      jSModuleArray0[2] = jSModule0;
      jSModuleArray0[3] = jSModule0;
      jSModuleArray0[4] = jSModule0;
      jSModuleArray0[5] = jSModule0;
      jSModuleArray0[6] = jSModule0;
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile(jSSourceFileArray0, jSModuleArray0, compilerOptions0);
      assertEquals(1, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("wnQ[CPe'lh!%\"<pQ,[3");
      JSModule jSModule0 = new JSModule("e+ +l<zatQ");
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[1];
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("GuCHY_K Q`CU&''G", "2-ZgL>.Hth");
      jSSourceFileArray0[0] = jSSourceFile0;
      JSModule[] jSModuleArray0 = new JSModule[7];
      jSModuleArray0[0] = jSModule0;
      jSModuleArray0[1] = jSModule0;
      jSModuleArray0[2] = jSModule0;
      jSModuleArray0[3] = jSModule0;
      jSModuleArray0[4] = jSModule0;
      jSModuleArray0[5] = jSModule0;
      jSModule0.add(jSSourceFile0);
      jSModuleArray0[6] = jSModule0;
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      compiler0.compile(jSSourceFileArray0, jSModuleArray0, compilerOptions0);
      assertEquals(3, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Enumeration<PipedInputStream> enumeration0 = (Enumeration<PipedInputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromInputStream("{PkF1", (InputStream) sequenceInputStream0);
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      compiler0.parseInputs();
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[0];
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
  public void test52()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("", (Charset) null);
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      // Undeclared exception!
      try { 
        compiler0.toSourceArray();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: java.lang.NullPointerException
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      DefaultPassConfig defaultPassConfig0 = new DefaultPassConfig(compilerOptions0);
      compiler0.setPassConfig(defaultPassConfig0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      PassConfig passConfig0 = compiler0.getPassConfig();
      PassConfig.PassConfigDelegate passConfig_PassConfigDelegate0 = new PassConfig.PassConfigDelegate(passConfig0);
      // Undeclared exception!
      try { 
        compiler0.setPassConfig(passConfig_PassConfigDelegate0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // this.passes has already been assigned
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
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
  public void test56()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.initOptions(compilerOptions0);
      compiler0.startPass((String) null);
      // Undeclared exception!
      try { 
        compiler0.startPass((String) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.reportCodeChange();
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[0];
      JSModule[] jSModuleArray0 = new JSModule[0];
      Result result0 = compiler0.compile(jSSourceFileArray0, jSModuleArray0, compilerOptions0);
      assertFalse(result0.success);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      CompilerOptions.TracerMode compilerOptions_TracerMode0 = CompilerOptions.TracerMode.ALL;
      compilerOptions0.tracer = compilerOptions_TracerMode0;
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("");
      // Undeclared exception!
      try { 
        compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("vPvJIPxAa}3B}sVD");
      boolean boolean0 = compiler0.areNodesEqualForInlining(node0, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("Eml8`/q8=B");
      // Undeclared exception!
      try { 
        compiler0.newExternInput("Eml8`/q8=B");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("", (Charset) null);
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      // Undeclared exception!
      try { 
        compiler0.newExternInput("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Conflicting externs name: 
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("+(");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("com.google.common.collect.ImmutableSetMultimap$BuilderMultimap");
      JsAst jsAst0 = new JsAst(jSSourceFile0);
      compiler0.addIncrementalSourceAst(jsAst0);
      // Undeclared exception!
      try { 
        compiler0.addIncrementalSourceAst(jsAst0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Duplicate input of name com.google.common.collect.ImmutableSetMultimap$BuilderMultimap
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      ReverseAbstractInterpreter reverseAbstractInterpreter0 = compiler0.getReverseAbstractInterpreter();
      ReverseAbstractInterpreter reverseAbstractInterpreter1 = compiler0.getReverseAbstractInterpreter();
      assertSame(reverseAbstractInterpreter1, reverseAbstractInterpreter0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("", (Charset) null);
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      assertEquals(1, compiler0.getErrorCount());
      
      compiler0.parseInputs();
      compiler0.parseInputs();
      assertEquals(2, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getColumnIndex();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      boolean boolean0 = compiler_CodeBuilder0.endsWith("#oC(#9'~|.");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      Compiler.CodeBuilder compiler_CodeBuilder1 = compiler_CodeBuilder0.append("generateReport");
      boolean boolean0 = compiler_CodeBuilder1.endsWith("Eml8`/q8=B");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      compiler_CodeBuilder0.append("Array");
      boolean boolean0 = compiler_CodeBuilder0.endsWith("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("vPvJIPxAa}3B}sVD");
      compiler0.optimize();
      assertFalse(compiler0.isIdeMode());
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[2];
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("configurable");
      jSSourceFileArray0[0] = jSSourceFile0;
      jSSourceFileArray0[1] = jSSourceFileArray0[0];
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile(jSSourceFileArray0, jSSourceFileArray0, compilerOptions0);
      boolean boolean0 = compiler0.isInliningForbidden();
      assertEquals(2, compiler0.getErrorCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("vPvJIPxAa}3B}sVD");
      Node node0 = compiler0.parseTestCode("vPvJIPxAa}3B}sVD");
      assertEquals(1, Node.PROPERTY_FLAG);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      DefaultPassConfig defaultPassConfig0 = new DefaultPassConfig(compilerOptions0);
      defaultPassConfig0.getPassGraph();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("// Input %num%", (Charset) null);
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      assertEquals(1, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.throwInternalError("Deq2qv==3_", (Exception) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // Deq2qv==3_
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Region region0 = compiler0.getSourceRegion("XOf=Cv;bz,f3t,", 0);
      assertNull(region0);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("HL'n(Ot0z|^CzUVQU*");
      Region region0 = compiler0.getSourceRegion("HL'n(Ot0z|^CzUVQU*", 4);
      assertNull(region0);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode(" [testcode] ");
      Region region0 = compiler0.getSourceRegion(" [testcode] ", 47);
      assertFalse(compiler0.hasErrors());
      assertNull(region0);
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSModule jSModule0 = new JSModule("");
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
  public void test78()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("", (Charset) null);
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      assertEquals(1, compiler0.getErrorCount());
      
      compiler0.getNodeForCodeInsertion((JSModule) null);
      assertEquals(2, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[0];
      JSModule[] jSModuleArray0 = new JSModule[0];
      compiler0.compile(jSSourceFileArray0, jSModuleArray0, compilerOptions0);
      // Undeclared exception!
      try { 
        compiler0.getNodeForCodeInsertion((JSModule) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No inputs
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      String string0 = compiler0.getAstDotGraph();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[0];
      JSModule[] jSModuleArray0 = new JSModule[0];
      compiler0.compile(jSSourceFileArray0, jSModuleArray0, compilerOptions0);
      compiler0.parseInputs();
      String string0 = compiler0.getAstDotGraph();
      assertEquals("digraph AST {\n  node [color=lightblue2, style=filled];\n  node0 [label=\"BLOCK\"];\n  node0 -> RETURN [label=\"UNCOND\", fontcolor=\"red\", weight=0.01, color=\"red\"];\n}\n", string0);
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      ErrorManager errorManager0 = compiler0.getErrorManager();
      assertEquals(0, errorManager0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ErrorManager errorManager0 = compiler0.getErrorManager();
      assertEquals(0, errorManager0.getErrorCount());
  }
}
