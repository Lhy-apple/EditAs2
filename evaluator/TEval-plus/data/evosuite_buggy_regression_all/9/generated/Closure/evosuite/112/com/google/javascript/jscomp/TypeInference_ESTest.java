/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:03:13 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.collect.ImmutableSortedMap;
import com.google.javascript.jscomp.ClosureCodingConvention;
import com.google.javascript.jscomp.CodingConvention;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ControlFlowGraph;
import com.google.javascript.jscomp.GoogleCodingConvention;
import com.google.javascript.jscomp.LinkedFlowScope;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.jscomp.TypeInference;
import com.google.javascript.jscomp.type.ClosureReverseAbstractInterpreter;
import com.google.javascript.jscomp.type.FlowScope;
import com.google.javascript.jscomp.type.ReverseAbstractInterpreter;
import com.google.javascript.jscomp.type.SemanticReverseAbstractInterpreter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.BooleanLiteralSet;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeInference_ESTest extends TypeInference_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("LB:$~'");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("LB:$~'", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("`Lj:bh$R!'>", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("`Lj:bh$R!'>", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("LBj:bh$~'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("Expected compiler to call an error manager: ", "$>$");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("`Lj:bh$R!'>", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("LB:$~'");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("`Lj:bh$R!'>", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("eO");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("</ul>", "eO");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascr6pt.jsXomp.NodeTraveraal$AbstructShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstactShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscom.NodeTraversal$AbstractShallowStatementCallback", "com.google.javascript.jscom.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("LB:$~'");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("`Lj:bh$R!'>", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascr6pt.jsXomp.NodeTraveraal$AbstructShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("~");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("~", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("LB:$~'");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("LB:$~'", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("`Lj:bh$R!'>", "com.google.javascript.jscomp.TypeInference$BooleanOutcomePair");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("o>;yB#dim>YF", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("~");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("~", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("`Lj:bh$R!'>", "com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("`Lj:bh$R!'>");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("`Lj:bh$R!'>", "ANNOTATION");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Multiple entries with same key: constructor=NOT_IMPLEMENTED and constructor=CONSTRUCTOR
         //
         verifyException("com.google.common.collect.ImmutableMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      BooleanLiteralSet booleanLiteralSet0 = BooleanLiteralSet.BOTH;
      BooleanLiteralSet booleanLiteralSet1 = TypeInference.getBooleanOutcomes(booleanLiteralSet0, booleanLiteralSet0, true);
      assertSame(booleanLiteralSet1, booleanLiteralSet0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      BooleanLiteralSet booleanLiteralSet0 = BooleanLiteralSet.FALSE;
      BooleanLiteralSet booleanLiteralSet1 = TypeInference.getBooleanOutcomes(booleanLiteralSet0, booleanLiteralSet0, false);
      assertSame(booleanLiteralSet0, booleanLiteralSet1);
  }
}