/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:03:39 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AmbiguateProperties;
import com.google.javascript.jscomp.Compiler;
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
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.DefaultPassConfig$15");
      ambiguateProperties0.process(node0, node0);
      assertEquals(30, Node.SKIP_INDEXES_PROP);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      char[] charArray0 = new char[1];
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, charArray0);
      Map<String, String> map0 = ambiguateProperties0.getRenamingMap();
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      Node node0 = new Node(40, 40, 40);
      Node node1 = new Node(64, node0, 37, 29);
      // Undeclared exception!
      try { 
        ambiguateProperties0.process(node1, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // String node not created with Node.newString
         //
         verifyException("com.google.javascript.rhino.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      Node node0 = new Node(64);
      Node node1 = new Node(64, node0, 37, 29);
      // Undeclared exception!
      try { 
        ambiguateProperties0.process(node1, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      Node node0 = new Node(36);
      Node node1 = new Node(35, node0, 32, 1);
      ambiguateProperties0.process(node0, node1);
      assertEquals(46, Node.IS_NAMESPACE);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      Node node0 = new Node(40, 40, 40);
      Node node1 = new Node(64, node0, 37, 29);
      // Undeclared exception!
      try { 
        ambiguateProperties0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // String node not created with Node.newString
         //
         verifyException("com.google.javascript.rhino.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      Node node0 = new Node(64, 64, 64);
      Node node1 = new Node(64, node0, 37, 29);
      // Undeclared exception!
      try { 
        ambiguateProperties0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      Node node0 = new Node(40);
      Node node1 = new Node(35, node0, 32, 2);
      // Undeclared exception!
      try { 
        ambiguateProperties0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // String node not created with Node.newString
         //
         verifyException("com.google.javascript.rhino.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.DefaultPassConfig$15");
      Node node1 = new Node(12);
      ambiguateProperties0.process(node1, node0);
      ambiguateProperties0.process(node1, node0);
      assertEquals(38, Node.SOURCEFILE_PROP);
  }
}
