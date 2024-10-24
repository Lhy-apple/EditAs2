/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:56:11 GMT 2023
 */

package org.apache.commons.jxpath.ri.compiler;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Iterator;
import java.util.Locale;
import org.apache.commons.jxpath.BasicNodeSet;
import org.apache.commons.jxpath.BasicVariables;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.ri.EvalContext;
import org.apache.commons.jxpath.ri.JXPathContextReferenceImpl;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.axes.AncestorContext;
import org.apache.commons.jxpath.ri.axes.NamespaceContext;
import org.apache.commons.jxpath.ri.axes.NodeSetContext;
import org.apache.commons.jxpath.ri.axes.PredicateContext;
import org.apache.commons.jxpath.ri.axes.RootContext;
import org.apache.commons.jxpath.ri.compiler.Constant;
import org.apache.commons.jxpath.ri.compiler.CoreOperationEqual;
import org.apache.commons.jxpath.ri.compiler.CoreOperationNotEqual;
import org.apache.commons.jxpath.ri.compiler.CoreOperationSubtract;
import org.apache.commons.jxpath.ri.compiler.CoreOperationUnion;
import org.apache.commons.jxpath.ri.compiler.Expression;
import org.apache.commons.jxpath.ri.compiler.NameAttributeTest;
import org.apache.commons.jxpath.ri.compiler.NodeNameTest;
import org.apache.commons.jxpath.ri.compiler.VariableReference;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.VariablePointer;
import org.apache.commons.jxpath.ri.model.beans.BeanPointer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CoreOperationCompare_ESTest extends CoreOperationCompare_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Constant constant0 = new Constant("' for path: ");
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(constant0, constant0);
      CoreOperationSubtract coreOperationSubtract0 = new CoreOperationSubtract(coreOperationEqual0, coreOperationEqual0);
      boolean boolean0 = coreOperationEqual0.equal((EvalContext) null, coreOperationSubtract0, coreOperationSubtract0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Constant constant0 = new Constant(".Uuhv=G8k");
      QName qName0 = new QName(".Uuhv=G8k");
      JXPathContextReferenceImpl jXPathContextReferenceImpl0 = (JXPathContextReferenceImpl)JXPathContext.newContext((Object) constant0);
      Locale locale0 = new Locale(".Uuhv=G8k", ".Uuhv=G8k");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      AncestorContext ancestorContext0 = new AncestorContext((EvalContext) null, true, nodeNameTest0);
      BeanPointer beanPointer0 = (BeanPointer)NodePointer.newNodePointer(qName0, ancestorContext0, locale0);
      RootContext rootContext0 = new RootContext(jXPathContextReferenceImpl0, beanPointer0);
      VariableReference variableReference0 = new VariableReference(qName0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(variableReference0, variableReference0);
      // Undeclared exception!
      try { 
        nameAttributeTest0.equal(rootContext0, variableReference0, constant0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Undefined variable: .Uuhv=G8k
         //
         verifyException("org.apache.commons.jxpath.ri.model.VariablePointer$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Constant constant0 = new Constant("&b");
      QName qName0 = new QName("&b");
      JXPathContextReferenceImpl jXPathContextReferenceImpl0 = (JXPathContextReferenceImpl)JXPathContext.newContext((Object) constant0);
      Locale locale0 = new Locale("&b", "&b");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      AncestorContext ancestorContext0 = new AncestorContext((EvalContext) null, true, nodeNameTest0);
      BeanPointer beanPointer0 = (BeanPointer)NodePointer.newNodePointer(qName0, ancestorContext0, locale0);
      RootContext rootContext0 = new RootContext(jXPathContextReferenceImpl0, beanPointer0);
      PredicateContext predicateContext0 = new PredicateContext(rootContext0, constant0);
      VariableReference variableReference0 = new VariableReference(qName0);
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(constant0, constant0);
      boolean boolean0 = coreOperationNotEqual0.equal(predicateContext0, variableReference0, variableReference0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Constant constant0 = new Constant("&b");
      QName qName0 = new QName("&b");
      JXPathContextReferenceImpl jXPathContextReferenceImpl0 = (JXPathContextReferenceImpl)JXPathContext.newContext((Object) constant0);
      Locale locale0 = new Locale("&b", "&b");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      AncestorContext ancestorContext0 = new AncestorContext((EvalContext) null, true, nodeNameTest0);
      BeanPointer beanPointer0 = (BeanPointer)NodePointer.newNodePointer(qName0, ancestorContext0, locale0);
      RootContext rootContext0 = new RootContext(jXPathContextReferenceImpl0, beanPointer0);
      VariableReference variableReference0 = new VariableReference(qName0);
      Expression[] expressionArray0 = new Expression[3];
      expressionArray0[0] = (Expression) variableReference0;
      expressionArray0[1] = (Expression) variableReference0;
      expressionArray0[2] = (Expression) constant0;
      CoreOperationUnion coreOperationUnion0 = new CoreOperationUnion(expressionArray0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(coreOperationUnion0, coreOperationUnion0);
      Iterator iterator0 = nameAttributeTest0.iterate(rootContext0);
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Constant constant0 = new Constant("b");
      QName qName0 = new QName("b", "b");
      JXPathContextReferenceImpl jXPathContextReferenceImpl0 = (JXPathContextReferenceImpl)JXPathContext.newContext((Object) constant0);
      Locale locale0 = new Locale("b", "b");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "b");
      AncestorContext ancestorContext0 = new AncestorContext((EvalContext) null, true, nodeNameTest0);
      BeanPointer beanPointer0 = (BeanPointer)NodePointer.newNodePointer(qName0, ancestorContext0, locale0);
      RootContext rootContext0 = new RootContext(jXPathContextReferenceImpl0, beanPointer0);
      Expression[] expressionArray0 = new Expression[0];
      CoreOperationUnion coreOperationUnion0 = new CoreOperationUnion(expressionArray0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(coreOperationUnion0, constant0);
      Iterator iterator0 = nameAttributeTest0.iterate(rootContext0);
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Constant constant0 = new Constant(".");
      RootContext rootContext0 = new RootContext((JXPathContextReferenceImpl) null, (NodePointer) null);
      Expression[] expressionArray0 = new Expression[0];
      CoreOperationUnion coreOperationUnion0 = new CoreOperationUnion(expressionArray0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(constant0, coreOperationUnion0);
      Iterator iterator0 = nameAttributeTest0.iterate(rootContext0);
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Constant constant0 = new Constant("pHOo`dYI@SZB'-Z");
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(constant0, constant0);
      PredicateContext predicateContext0 = new PredicateContext((EvalContext) null, coreOperationNotEqual0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(constant0, coreOperationNotEqual0);
      Iterator iterator0 = nameAttributeTest0.iterate(predicateContext0);
      boolean boolean0 = nameAttributeTest0.contains(iterator0, "pHOo`dYI@SZB'-Z");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Constant constant0 = new Constant("p8`o`Y@SZBI-Z");
      QName qName0 = new QName("p8`o`Y@SZBI-Z");
      BasicNodeSet basicNodeSet0 = new BasicNodeSet();
      NodeSetContext nodeSetContext0 = new NodeSetContext((EvalContext) null, basicNodeSet0);
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      NamespaceContext namespaceContext0 = new NamespaceContext(nodeSetContext0, nodeNameTest0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(constant0, constant0);
      Iterator iterator0 = nameAttributeTest0.iterate((EvalContext) null);
      boolean boolean0 = nameAttributeTest0.findMatch(namespaceContext0, iterator0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Constant constant0 = new Constant("n9fbL~bO9^ro\"");
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(constant0, constant0);
      BasicVariables basicVariables0 = new BasicVariables();
      VariablePointer variablePointer0 = new VariablePointer(basicVariables0, (QName) null);
      NodePointer nodePointer0 = NodePointer.newChildNodePointer((NodePointer) null, (QName) null, variablePointer0);
      // Undeclared exception!
      try { 
        coreOperationEqual0.equal(nodePointer0, variablePointer0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.VariablePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Constant constant0 = new Constant("pHOo`dYI@SZB'-Z");
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(constant0, constant0);
      PredicateContext predicateContext0 = new PredicateContext((EvalContext) null, coreOperationNotEqual0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(constant0, coreOperationNotEqual0);
      Iterator iterator0 = nameAttributeTest0.iterate(predicateContext0);
      boolean boolean0 = nameAttributeTest0.contains(iterator0, (Object) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Constant constant0 = new Constant((String) null);
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(constant0, constant0);
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(constant0, constant0);
      boolean boolean0 = coreOperationNotEqual0.equal((Object) null, coreOperationEqual0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Constant constant0 = new Constant("pHOo`dYI@SZB'-Z");
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(constant0, constant0);
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(coreOperationEqual0, constant0);
      Double double0 = Expression.ONE;
      boolean boolean0 = coreOperationNotEqual0.equal(coreOperationEqual0, double0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Constant constant0 = new Constant("p8`o`Y@SZBI-Z");
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(constant0, constant0);
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(coreOperationEqual0, constant0);
      boolean boolean0 = coreOperationNotEqual0.equal("p8`o`Y@SZBI-Z", coreOperationEqual0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Constant constant0 = new Constant("%0]vM{v~#B");
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(constant0, constant0);
      boolean boolean0 = coreOperationEqual0.equal(constant0, "%0]vM{v~#B");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Constant constant0 = new Constant("q !D,");
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(constant0, constant0);
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(coreOperationEqual0, constant0);
      boolean boolean0 = coreOperationNotEqual0.equal(coreOperationEqual0, constant0);
      assertFalse(boolean0);
  }
}
