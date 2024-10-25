/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:37:27 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AmbiguateProperties;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.JsAst;
import com.google.javascript.rhino.Node;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AmbiguateProperties_ESTest extends AmbiguateProperties_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      char[] charArray0 = new char[1];
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, charArray0);
      Node node0 = new Node((-1221), (-1221), (-1221));
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("com.google.javascript.jscomp.IgnoreCajaProperties$Traversal", "com.google.javascript.jscomp.IgnoreCajaProperties$Traversal");
      JsAst jsAst0 = new JsAst(jSSourceFile0);
      Node node1 = jsAst0.getAstRoot(compiler0);
      ambiguateProperties0.process(node0, node1);
      ambiguateProperties0.process(node0, node1);
      assertEquals(36, Node.OPT_ARG_NAME);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      char[] charArray0 = new char[3];
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, charArray0);
      Map<String, String> map0 = ambiguateProperties0.getRenamingMap();
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      char[] charArray0 = new char[28];
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, charArray0);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("com.google.javascript.jscomp.IgnoreCajaProperties$Traversal", "com.google.javascript.jscomp.IgnoreCajaProperties$Traversal");
      JsAst jsAst0 = new JsAst(jSSourceFile0);
      Node node0 = jsAst0.getAstRoot(compiler0);
      ambiguateProperties0.process(node0, node0);
      assertEquals(13, Node.CASES_PROP);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      char[] charArray0 = new char[0];
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, charArray0);
      Node node0 = new Node(64, 64, 64);
      Node node1 = new Node(64, node0);
      // Undeclared exception!
      try { 
        ambiguateProperties0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      char[] charArray0 = new char[4];
      Node node0 = Node.newString("");
      Node node1 = new Node(35, node0, 1, (-606));
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, charArray0);
      ambiguateProperties0.process(node1, node1);
      assertFalse(node1.isVarArgs());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      char[] charArray0 = new char[0];
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, charArray0);
      Node node0 = new Node(64, 64, 64);
      Node node1 = new Node(64, node0);
      // Undeclared exception!
      try { 
        ambiguateProperties0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      char[] charArray0 = new char[18];
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, charArray0);
      Node node0 = new Node('\u0000', '\u0000', '\u0000');
      Node node1 = new Node(35, node0, 24, 1912);
      ambiguateProperties0.process(node1, node1);
      assertNotSame(node0, node1);
  }
}
