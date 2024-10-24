/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:01:26 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.JsAst;
import com.google.javascript.jscomp.SourceFile;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsAst_ESTest extends JsAst_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile("VF=fEn>Y9Yvul+:");
      JsAst jsAst0 = new JsAst(sourceFile0);
      jsAst0.clearAst();
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile("duF2qC>8)U%Y LzhD");
      JsAst jsAst0 = new JsAst(sourceFile0);
      jsAst0.setSourceFile(sourceFile0);
      assertEquals("duF2qC>8)U%Y LzhD", sourceFile0.getOriginalPath());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile("t$]F[<{kG.q:F2z5");
      JsAst jsAst0 = new JsAst(sourceFile0);
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      compiler0.compile(sourceFile0, sourceFile0, compilerOptions0);
      jsAst0.getAstRoot(compiler0);
      jsAst0.getAstRoot(compiler0);
      assertEquals(2, compiler0.getErrorCount());
  }
}
