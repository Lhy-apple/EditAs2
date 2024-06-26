/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:14:52 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.JsAst;
import com.google.javascript.jscomp.SourceFile;
import com.google.javascript.jscomp.SyntheticAst;
import com.google.javascript.rhino.Node;
import java.io.File;
import java.nio.charset.Charset;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsAst_ESTest extends JsAst_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Charset charset0 = Charset.defaultCharset();
      SourceFile sourceFile0 = SourceFile.fromFile(".2--vD)O,0z~p5", charset0);
      JsAst jsAst0 = new JsAst(sourceFile0);
      jsAst0.clearAst();
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      SyntheticAst syntheticAst0 = new SyntheticAst("UP#;BD2Y'l{i'JS");
      SourceFile sourceFile0 = syntheticAst0.getSourceFile();
      JsAst jsAst0 = new JsAst(sourceFile0);
      jsAst0.setSourceFile(sourceFile0);
      assertEquals("UP#;BD2Y'l{i'JS", sourceFile0.getName());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      File file0 = MockFile.createTempFile(" XS.t`Wx8!pU", " XS.t`Wx8!pU");
      SourceFile sourceFile0 = SourceFile.fromFile(file0);
      JsAst jsAst0 = new JsAst(sourceFile0);
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("wk]7wE[N\";v");
      jsAst0.getAstRoot(compiler0);
      Node node0 = jsAst0.getAstRoot(compiler0);
      assertNotNull(node0);
      assertEquals((-1), node0.getCharno());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      SourceFile.fromCode("cA", "cA");
      File file0 = MockFile.createTempFile(" XS.t`Wx8!pU", " XS.t`Wx8!pU");
      SourceFile.fromFile(file0);
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("cA");
      assertEquals(0, node0.getCharno());
      assertNotNull(node0);
  }
}
