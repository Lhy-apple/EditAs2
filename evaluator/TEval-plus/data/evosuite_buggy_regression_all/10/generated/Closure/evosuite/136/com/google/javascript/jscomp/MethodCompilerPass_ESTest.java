/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:53:55 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.InlineGetters;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MethodCompilerPass_ESTest extends MethodCompilerPass_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      InlineGetters inlineGetters0 = new InlineGetters(compiler0);
      // Undeclared exception!
      try { 
        inlineGetters0.process((Node) null, (Node) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.goo=le.jaacript.jNcomp.MethodCompilerPass$");
      Node node1 = Node.newString("com.goo=le.jaacript.jNcomp.MethodCompilerPass$");
      node1.addChildToBack(node0);
      InlineGetters inlineGetters0 = new InlineGetters(compiler0);
      // Undeclared exception!
      try { 
        inlineGetters0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(SCRIPT):  [testcode] :1:0
         // [source unknown]
         //   Parent(STRING com.goo=le.jaacript.jNcomp.MethodCompilerPass$):  [testcode] :-1:-1
         // [source unknown]
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(35);
      InlineGetters inlineGetters0 = new InlineGetters(compiler0);
      // Undeclared exception!
      try { 
        inlineGetters0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("8[B$J2AIG8IaJMgR$4");
      Node node1 = Node.newString(1538, "8[B$J2AIG8IaJMgR$4");
      Node node2 = new Node(33, node1, node1, node0, node0, 16, 1);
      InlineGetters inlineGetters0 = new InlineGetters(compiler0);
      // Undeclared exception!
      try { 
        inlineGetters0.process(node2, node1);
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
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(64);
      Node node1 = Node.newString("Vh91@U%Q0nHp(;", 12, 12);
      node0.addChildToBack(node1);
      InlineGetters inlineGetters0 = new InlineGetters(compiler0);
      // Undeclared exception!
      try { 
        inlineGetters0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      InlineGetters inlineGetters0 = new InlineGetters(compiler0);
      Node node0 = Node.newNumber((double) 64, 64, 64);
      Node node1 = new Node(64, node0, node0, node0);
      // Undeclared exception!
      try { 
        inlineGetters0.process(node1, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(64);
      Node node1 = new Node(35, 40, 1555);
      Node node2 = new Node((-4546), node0, node0, node1, node0, 33, 11);
      InlineGetters inlineGetters0 = new InlineGetters(compiler0);
      // Undeclared exception!
      try { 
        inlineGetters0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }
}
