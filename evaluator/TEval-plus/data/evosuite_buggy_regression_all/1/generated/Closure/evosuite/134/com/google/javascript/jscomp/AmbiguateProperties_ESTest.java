/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:15:46 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AmbiguateProperties;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.ScriptOrFnNode;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AmbiguateProperties_ESTest extends AmbiguateProperties_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString("com.google.javascript.jscomp.SymbolTable");
      ScriptOrFnNode scriptOrFnNode0 = (ScriptOrFnNode)compiler0.parseSyntheticCode("com.google.javascript.jscomp.SymbolTable", "com.google.javascript.jscomp.SymbolTable");
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      ambiguateProperties0.process(node0, scriptOrFnNode0);
      ambiguateProperties0.process(node0, scriptOrFnNode0);
      assertEquals(1, ScriptOrFnNode.NO_DUPLICATE);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      Map<String, String> map0 = ambiguateProperties0.getRenamingMap();
      assertTrue(map0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.AmbiguateProperies$PropertyGraph", "com.google.javascript.jscomp.AmbiguateProperies$PropertyGraph");
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      ambiguateProperties0.process(node0, node0);
      assertEquals(2, Node.ATTRIBUTE_FLAG);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(64);
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      ambiguateProperties0.process(node0, node0);
      assertEquals(42, Node.NO_SIDE_EFFECTS_CALL);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(64);
      Node node1 = new Node(1);
      node0.addChildToBack(node1);
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      // Undeclared exception!
      try { 
        ambiguateProperties0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(64);
      Node node1 = new Node(29);
      node0.addChildToBack(node1);
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      // Undeclared exception!
      try { 
        ambiguateProperties0.process(node1, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(35);
      Node node1 = new Node((-5469));
      node0.addChildToBack(node1);
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      ambiguateProperties0.process(node1, node0);
      assertEquals(7, Node.LOCAL_PROP);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      char[] charArray0 = new char[7];
      Node node0 = Node.newString("+", 1408, 1408);
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, charArray0);
      Node node1 = Node.newString(35, "+");
      node1.addChildrenToFront(node0);
      ambiguateProperties0.process(node0, node1);
      assertEquals(37, Node.SYNTHETIC_BLOCK_PROP);
  }
}
