/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:19:35 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AmbiguateProperties;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.List;
import java.util.Map;
import java.util.Vector;
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
      Map<String, String> map0 = ambiguateProperties0.getRenamingMap();
      assertTrue(map0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.AmbiguateProperties$1");
      ambiguateProperties0.process(node0, node0);
      assertEquals(1, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(64);
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      ambiguateProperties0.process(node0, node0);
      assertEquals(30, Node.SKIP_INDEXES_PROP);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newNumber((double) 64, 64, 64);
      Node node1 = new Node(64, node0, node0, 31, 1);
      char[] charArray0 = new char[1];
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, charArray0);
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
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      Vector<JSType> vector0 = new Vector<JSType>(35, 35);
      Node node0 = jSTypeRegistry0.createParameters((List<JSType>) vector0);
      Node node1 = new Node(35, node0);
      ambiguateProperties0.process(node1, node1);
      assertTrue(node1.hasChildren());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newNumber((double) 64, 64, 64);
      Node node1 = new Node(64, node0, node0, 31, 1);
      char[] charArray0 = new char[1];
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, charArray0);
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
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      Node node0 = Node.newString("com.google.javascript.jscomp.AmbiguateProperties$1", 35, 35);
      Node node1 = new Node(35, node0, node0, node0);
      ambiguateProperties0.process(node0, node1);
      assertEquals(2, Node.POST_FLAG);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AmbiguateProperties ambiguateProperties0 = new AmbiguateProperties(compiler0, (char[]) null);
      Node node0 = Node.newString("com.googl.jaPcript.NoDp.AbutePrPperties$1");
      Node node1 = compiler0.parseTestCode("com.googl.jaPcript.NoDp.AbutePrPperties$1");
      ambiguateProperties0.process(node0, node1);
      ambiguateProperties0.process(node0, node1);
      assertFalse(node0.hasChildren());
  }
}