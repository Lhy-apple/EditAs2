/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:34:03 GMT 2023
 */

package com.google.javascript.jscomp.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.ClosureCodingConvention;
import com.google.javascript.jscomp.GoogleCodingConvention;
import com.google.javascript.jscomp.JqueryCodingConvention;
import com.google.javascript.jscomp.type.ClosureReverseAbstractInterpreter;
import com.google.javascript.jscomp.type.FlowScope;
import com.google.javascript.jscomp.type.SemanticReverseAbstractInterpreter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.StaticSlot;
import java.util.LinkedList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SemanticReverseAbstractInterpreter_ESTest extends SemanticReverseAbstractInterpreter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      Node node0 = new Node(26);
      // Undeclared exception!
      try { 
        semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node0, (FlowScope) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.type.SemanticReverseAbstractInterpreter", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("HK");
      Node node1 = new Node(15, node0, node0, node0, node0, 15, 0);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, true);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("");
      Node node1 = new Node(13, node0, node0, node0, node0, (-43), 47);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, false);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("Ea5p!%9_'4-LYFC");
      Node node1 = new Node(12, node0, node0, node0, node0, 4095, 36);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, false);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("com.google.javascript.jscomp.type.SemanticReverseAbstractInterpreter$3");
      Node node1 = new Node(45, node0, node0, node0, node0);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, true);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("Ea5pG%9L'4LYF");
      Node node1 = new Node(45, node0, node0, node0, node0, 40, 29);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, false);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Node node0 = Node.newString("%E5c5!/%9_'4-^YF ");
      Node node1 = new Node(46, node0, node0, node0, node0, 51, 37);
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(jqueryCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, false);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("com.google.javascript.rhino.jstype.AllType");
      Node node1 = new Node(14, node0, node0, node0, node0, 14, 54);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, false);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("Ea5pG%9L'4LYF");
      Node node1 = new Node(16, node0, node0, node0, node0, 4095, 16);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.firstPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, true);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("hEac5!/%9_'4-^YF ");
      Node node1 = new Node(17, node0, node0, node0, node0, 40, 29);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, true);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("ZvnD%k?g}#M#<+lL]Y'");
      Node node1 = new Node(26, node0, node0, node0, node0, 32, 4095);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, jSTypeRegistry0);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.firstPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, false);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("hEac5!/%9_'4-^YF ");
      Node node1 = new Node(33, node0, node0, node0, node0, 40, 29);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, true);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Node node0 = Node.newString(",");
      Node node1 = new Node(38, node0, node0, node0, node0);
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(jqueryCodingConvention0, (JSTypeRegistry) null);
      // Undeclared exception!
      try { 
        semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, true);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // NAME is not a string node
         //
         verifyException("com.google.javascript.rhino.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Node node0 = Node.newString(",");
      Node node1 = new Node(51, node0, node0, node0, node0, 4095, 41);
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(jqueryCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, true);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("'\"r<fPx#z{8x6V)-");
      Node node1 = new Node(52, node0, node0, node0, node0, 15, 45);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, false);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("hEa55!/%9_'4-DYF ");
      Node node1 = new Node(86, node0, node0, node0, node0, 146, 38);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      // Undeclared exception!
      try { 
        semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.type.SemanticReverseAbstractInterpreter", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, jSTypeRegistry0);
      Node node0 = Node.newString("}uacG%9_M$-,Yr ");
      Node node1 = new Node(100, node0, node0, node0, node0);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, false);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("hEac5!/%9_'4-^YF ");
      Node node1 = new Node(101, node0, node0, node0, node0, 40, 49);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, true);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, jSTypeRegistry0);
      Node node0 = Node.newString("hEac5!/%9_'4-^YF ");
      Node node1 = new Node(111, node0, node0, node0, node0);
      Node node2 = new Node(6811, node1, node1, node1, node1);
      ClosureReverseAbstractInterpreter closureReverseAbstractInterpreter0 = new ClosureReverseAbstractInterpreter(googleCodingConvention0, jSTypeRegistry0);
      FlowScope flowScope0 = mock(FlowScope.class, new ViolatedAssumptionAnswer());
      doReturn((String) null, (String) null).when(flowScope0).toString();
      FlowScope flowScope1 = closureReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node0, flowScope0, false);
      FlowScope flowScope2 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, flowScope1, false);
      assertSame(flowScope1, flowScope2);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("#1tVuKTlIT,7Q>x8OV");
      Node node1 = new Node(101, node0, node0, node0, node0, 40, 49);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      // Undeclared exception!
      try { 
        semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.type.SemanticReverseAbstractInterpreter", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(closureCodingConvention0, jSTypeRegistry0);
      Node node0 = Node.newString("Not declared as a constructor", 0, 0);
      Node node1 = new Node(100, node0, node0, node0, node0);
      FlowScope flowScope0 = mock(FlowScope.class, new ViolatedAssumptionAnswer());
      doReturn((StaticSlot) null).when(flowScope0).findUniqueRefinedSlot(any(com.google.javascript.jscomp.type.FlowScope.class));
      doReturn((String) null, (String) null, (String) null).when(flowScope0).toString();
      FlowScope flowScope1 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, flowScope0, true);
      assertSame(flowScope1, flowScope0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = Node.newString("Ea5p!%9_'4-LYFC");
      Node node1 = new Node(12, node0, node0, node0, node0, 4095, 36);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, true);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, jSTypeRegistry0);
      Node node0 = Node.newString("yS;n");
      Node node1 = new Node(13, node0, node0, node0, node0);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, true);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Node node0 = Node.newString("hEac5!/%9_'4-^YF ");
      Node node1 = new Node(46, node0, node0, node0, node0, 51, 37);
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(jqueryCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.firstPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, true);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Node node0 = Node.newString(",");
      Node node1 = new Node(51, node0, node0, node0, node0, 4095, 41);
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(jqueryCodingConvention0, (JSTypeRegistry) null);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, false);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, (JSTypeRegistry) null);
      Node node0 = Node.newNumber((double) 51, 51, 51);
      Node node1 = new Node(51, node0, node0, 4095, 2358);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, true);
      assertNull(flowScope0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      SemanticReverseAbstractInterpreter semanticReverseAbstractInterpreter0 = new SemanticReverseAbstractInterpreter(googleCodingConvention0, jSTypeRegistry0);
      Node node0 = Node.newString("hEac5!/%9_'4-^YF ");
      Node node1 = new Node(111, node0, node0, node0, node0);
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      Node node2 = jSTypeRegistry0.createParameters((List<JSType>) linkedList0);
      Node node3 = new Node(38, node1, node1, node2, node2);
      FlowScope flowScope0 = semanticReverseAbstractInterpreter0.getPreciserScopeKnowingConditionOutcome(node1, (FlowScope) null, true);
      assertNull(flowScope0);
  }
}
