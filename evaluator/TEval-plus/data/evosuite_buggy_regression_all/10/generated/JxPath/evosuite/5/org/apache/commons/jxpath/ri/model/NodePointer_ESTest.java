/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:02:50 GMT 2023
 */

package org.apache.commons.jxpath.ri.model;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Locale;
import org.apache.commons.jxpath.JXPathBasicBeanInfo;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.ri.NamespaceResolver;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.compiler.NodeNameTest;
import org.apache.commons.jxpath.ri.compiler.NodeTest;
import org.apache.commons.jxpath.ri.compiler.NodeTypeTest;
import org.apache.commons.jxpath.ri.compiler.ProcessingInstructionTest;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.VariablePointer;
import org.apache.commons.jxpath.ri.model.beans.BeanPointer;
import org.apache.commons.jxpath.ri.model.beans.NullPointer;
import org.apache.commons.jxpath.ri.model.beans.NullPropertyPointer;
import org.apache.commons.jxpath.ri.model.beans.PropertyPointer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NodePointer_ESTest extends NodePointer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      QName qName0 = new QName("INvEdj: ", "INvEdj: ");
      Locale locale0 = Locale.CANADA;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, qName0, locale0);
      Object object0 = new Object();
      JXPathContext jXPathContext0 = JXPathContext.newContext(object0);
      // Undeclared exception!
      try { 
        nodePointer0.createChild(jXPathContext0, qName0, Integer.MIN_VALUE);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot create an object for path //INvEdj: :INvEdj: [-2147483647], operation is not allowed for this type of node
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      boolean boolean0 = nullPropertyPointer0.isAttribute();
      assertFalse(boolean0);
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer0.getIndex());
      assertTrue(nullPropertyPointer0.isRoot());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, (QName) null, variablePointer0);
      assertNotNull(nodePointer0);
      assertTrue(variablePointer0.isRoot());
      
      nodePointer0.printPointerChain();
      assertFalse(nodePointer0.isRoot());
      assertFalse(nodePointer0.isAttribute());
      assertEquals(Integer.MIN_VALUE, nodePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Locale locale0 = Locale.ITALY;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, locale0, locale0);
      // Undeclared exception!
      try { 
        nodePointer0.createPath((JXPathContext) null, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.PropertyOwnerPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      nullPropertyPointer0.compareTo(nullPropertyPointer0);
      assertFalse(nullPropertyPointer0.isRoot());
      assertFalse(nullPropertyPointer0.isAttribute());
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      variablePointer0.namespaceIterator();
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
      assertFalse(variablePointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      QName qName0 = new QName("d/n");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      variablePointer0.getDefaultNamespaceURI();
      assertFalse(variablePointer0.isAttribute());
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      // Undeclared exception!
      try { 
        variablePointer0.getNodeValue();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Undefined variable: null
         //
         verifyException("org.apache.commons.jxpath.ri.model.VariablePointer$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) variablePointer0);
      // Undeclared exception!
      try { 
        variablePointer0.createAttribute(jXPathContext0, (QName) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot create an attribute for path $null/@null, operation is not allowed for this type of node
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NamespaceResolver namespaceResolver0 = new NamespaceResolver();
      variablePointer0.setNamespaceResolver(namespaceResolver0);
      variablePointer0.getNamespaceResolver();
      assertFalse(variablePointer0.isAttribute());
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      nullPropertyPointer0.remove();
      assertFalse(nullPropertyPointer0.isRoot());
      assertFalse(nullPropertyPointer0.isAttribute());
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, (QName) null, variablePointer0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) null);
      NodePointer nodePointer1 = nodePointer0.createPath(jXPathContext0);
      assertEquals(Integer.MIN_VALUE, nodePointer1.getIndex());
      assertFalse(nodePointer1.isAttribute());
      assertFalse(nodePointer1.isRoot());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      QName qName0 = new QName("JJ}");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      variablePointer0.getNamespaceURI();
      assertFalse(variablePointer0.isAttribute());
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      variablePointer0.namespacePointer("<<unknown namespace>>");
      assertFalse(variablePointer0.isAttribute());
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      NullPointer nullPointer0 = new NullPointer((QName) null, locale0);
      PropertyPointer propertyPointer0 = nullPointer0.getPropertyPointer();
      assertFalse(propertyPointer0.isCollection());
      
      propertyPointer0.setIndex(0);
      propertyPointer0.asPath();
      assertEquals(0, propertyPointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      QName qName0 = new QName("*");
      QName qName1 = new QName("*", "*");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName1);
      boolean boolean0 = variablePointer0.testNode(nodeNameTest0);
      assertFalse(variablePointer0.isAttribute());
      assertTrue(boolean0);
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) null);
      // Undeclared exception!
      try { 
        variablePointer0.getPointerByKey(jXPathContext0, "RRED(PW:{PB<I*J;AIT-IT", "RRED(PW:{PB<I*J;AIT-IT");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot find an element by key - no KeyManager has been specified
         //
         verifyException("org.apache.commons.jxpath.JXPathContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, locale0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) null);
      // Undeclared exception!
      try { 
        nodePointer0.createChild(jXPathContext0, (QName) null, Integer.MIN_VALUE, (Object) locale0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot create an object for path //null[-2147483647], operation is not allowed for this type of node
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      // Undeclared exception!
      try { 
        variablePointer0.getPointerByID((JXPathContext) null, "<<unknown namespace>>");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Locale locale0 = Locale.FRANCE;
      NullPointer nullPointer0 = new NullPointer((QName) null, locale0);
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(nullPointer0, (QName) null, nullPointer0);
      nodePointer0.getNamespaceResolver();
      assertFalse(nodePointer0.isRoot());
      assertEquals(Integer.MIN_VALUE, nullPointer0.getIndex());
      assertEquals(Integer.MIN_VALUE, nodePointer0.getIndex());
      assertFalse(nodePointer0.isAttribute());
      assertFalse(nullPointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      NodePointer nodePointer0 = nullPropertyPointer0.getParent();
      assertFalse(nullPropertyPointer0.isCollection());
      assertNull(nodePointer0);
      assertFalse(nullPropertyPointer0.isAttribute());
      assertFalse(nullPropertyPointer0.isRoot());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NodePointer nodePointer0 = variablePointer0.getImmediateValuePointer();
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nodePointer0);
      NodePointer nodePointer1 = nullPropertyPointer0.getParent();
      assertNotNull(nodePointer1);
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer0.getIndex());
      assertFalse(nullPropertyPointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      QName qName0 = new QName("", "");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NodePointer nodePointer0 = variablePointer0.getImmediateValuePointer();
      boolean boolean0 = nodePointer0.isRoot();
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
      assertFalse(boolean0);
      assertFalse(variablePointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      boolean boolean0 = nullPropertyPointer0.isRoot();
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer0.getIndex());
      assertFalse(nullPropertyPointer0.isAttribute());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      boolean boolean0 = variablePointer0.testNode(nodeTypeTest0);
      assertTrue(boolean0);
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
      assertFalse(variablePointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      // Undeclared exception!
      try { 
        variablePointer0.getValue();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Undefined variable: null
         //
         verifyException("org.apache.commons.jxpath.ri.model.VariablePointer$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Locale locale0 = Locale.GERMANY;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, locale0, locale0);
      assertNotNull(nodePointer0);
      
      boolean boolean0 = nodePointer0.isActual();
      assertTrue(boolean0);
      assertEquals(Integer.MIN_VALUE, nodePointer0.getIndex());
      assertFalse(nodePointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Locale locale0 = Locale.GERMANY;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, locale0, locale0);
      assertEquals(Integer.MIN_VALUE, nodePointer0.getIndex());
      
      nodePointer0.setIndex(0);
      boolean boolean0 = nodePointer0.isActual();
      assertEquals(0, nodePointer0.getIndex());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Locale locale0 = Locale.GERMANY;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, locale0, locale0);
      nodePointer0.setIndex((-51));
      boolean boolean0 = nodePointer0.isActual();
      assertEquals((-51), nodePointer0.getIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Locale locale0 = Locale.GERMANY;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, locale0, locale0);
      nodePointer0.setIndex(3);
      boolean boolean0 = nodePointer0.isActual();
      assertEquals(3, nodePointer0.getIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      QName qName0 = new QName("INTERj: ", "INTERj: ");
      Locale locale0 = Locale.CANADA;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, qName0, locale0);
      nodePointer0.getRootNode();
      Object object0 = nodePointer0.getRootNode();
      assertNotNull(object0);
      assertEquals(Integer.MIN_VALUE, nodePointer0.getIndex());
      assertFalse(nodePointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.getRootNode();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Undefined variable: null
         //
         verifyException("org.apache.commons.jxpath.ri.model.VariablePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      boolean boolean0 = variablePointer0.testNode((NodeTest) null);
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
      assertFalse(variablePointer0.isAttribute());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      NodeNameTest nodeNameTest0 = new NodeNameTest((QName) null, "<<unknown namespace>>");
      boolean boolean0 = nullPropertyPointer0.testNode(nodeNameTest0);
      assertFalse(boolean0);
      assertFalse(nullPropertyPointer0.isAttribute());
      assertFalse(nullPropertyPointer0.isRoot());
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NodeNameTest nodeNameTest0 = new NodeNameTest((QName) null, "<<unknown namespace>>");
      boolean boolean0 = variablePointer0.testNode(nodeNameTest0);
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
      assertFalse(variablePointer0.isAttribute());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("<<unknown namespace>>");
      boolean boolean0 = variablePointer0.testNode(processingInstructionTest0);
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
      assertFalse(boolean0);
      assertFalse(variablePointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      boolean boolean0 = variablePointer0.testNode(nodeTypeTest0);
      assertFalse(variablePointer0.isAttribute());
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      QName qName0 = new QName("org.apache.commons.jxpath.ri.model.NodePointer");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      boolean boolean0 = nullPropertyPointer0.testNode(nodeTypeTest0);
      assertFalse(boolean0);
      assertFalse(nullPropertyPointer0.isRoot());
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer0.getIndex());
      assertFalse(nullPropertyPointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      QName qName0 = new QName("dI");
      QName qName1 = new QName("dI", "dI");
      VariablePointer variablePointer0 = new VariablePointer(qName1);
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = variablePointer0.testNode(nodeNameTest0);
      assertTrue(boolean0);
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
      assertFalse(variablePointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      QName qName0 = new QName("INTERj: ");
      QName qName1 = new QName("INTERj: ");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName1, "INTERj: ");
      boolean boolean0 = variablePointer0.testNode(nodeNameTest0);
      assertTrue(boolean0);
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
      assertFalse(variablePointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      variablePointer0.getLocale();
      assertFalse(variablePointer0.isAttribute());
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Locale locale0 = Locale.GERMANY;
      NullPointer nullPointer0 = new NullPointer((QName) null, locale0);
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(nullPointer0, (QName) null, (Object) null);
      assertFalse(nodePointer0.isRoot());
      
      boolean boolean0 = nodePointer0.isLanguage("<<unknown namespace>>");
      assertEquals(Integer.MIN_VALUE, nullPointer0.getIndex());
      assertFalse(boolean0);
      assertEquals(Integer.MIN_VALUE, nodePointer0.getIndex());
      assertFalse(nodePointer0.isAttribute());
      assertFalse(nullPointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("<<unknown namespace>>");
      nullPropertyPointer0.childIterator(processingInstructionTest0, false, (NodePointer) null);
      assertFalse(nullPropertyPointer0.isAttribute());
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.attributeIterator((QName) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.BeanAttributeIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      boolean boolean0 = variablePointer0.isDefaultNamespace((String) null);
      assertEquals(Integer.MIN_VALUE, variablePointer0.getIndex());
      assertFalse(variablePointer0.isAttribute());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      nullPropertyPointer0.printPointerChain();
      assertFalse(nullPropertyPointer0.isAttribute());
      assertFalse(nullPropertyPointer0.isCollection());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      NullPointer nullPointer0 = new NullPointer((QName) null, locale0);
      PropertyPointer propertyPointer0 = nullPointer0.getPropertyPointer();
      assertFalse(propertyPointer0.isAttribute());
      
      propertyPointer0.setAttribute(true);
      propertyPointer0.asPath();
      assertTrue(propertyPointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, locale0, locale0);
      NodePointer nodePointer1 = NodePointer.newChildNodePointer(nodePointer0, (QName) null, nodePointer0);
      nodePointer1.setIndex(352);
      nodePointer1.toString();
      assertEquals(352, nodePointer1.getIndex());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      NodePointer nodePointer0 = nullPropertyPointer0.getImmediateValuePointer();
      assertEquals(Integer.MIN_VALUE, nodePointer0.getIndex());
      assertTrue(variablePointer0.isRoot());
      assertFalse(nodePointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      QName qName0 = new QName("INTERj: ");
      Locale locale0 = Locale.FRANCE;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, qName0, locale0);
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.compareTo(nodePointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot compare pointers that do not belong to the same tree: '$INTERj: ' and '/'
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      int int0 = nullPropertyPointer0.compareTo(nullPropertyPointer0);
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer0.getIndex());
      assertFalse(nullPropertyPointer0.isAttribute());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, (Object) null, locale0);
      NodePointer nodePointer1 = NodePointer.newChildNodePointer(nodePointer0, (QName) null, (Object) null);
      NodePointer nodePointer2 = NodePointer.newChildNodePointer(nodePointer1, (QName) null, (Object) null);
      int int0 = nodePointer0.compareTo(nodePointer2);
      assertEquals((-1), int0);
      assertEquals(Integer.MIN_VALUE, nodePointer2.getIndex());
      assertFalse(nodePointer0.isAttribute());
      assertEquals(Integer.MIN_VALUE, nodePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Locale locale0 = Locale.GERMAN;
      QName qName0 = new QName("V_");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, qName0, locale0);
      Class<Integer> class0 = Integer.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0, class0);
      Object object0 = new Object();
      BeanPointer beanPointer0 = new BeanPointer(nodePointer0, qName0, object0, jXPathBasicBeanInfo0);
      PropertyPointer propertyPointer0 = beanPointer0.getPropertyPointer();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      NodePointer nodePointer1 = NodePointer.newChildNodePointer(propertyPointer0, qName0, nodeNameTest0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nodePointer0);
      NodePointer nodePointer2 = nullPropertyPointer0.getValuePointer();
      int int0 = nodePointer1.compareTo(nodePointer2);
      assertEquals(Integer.MIN_VALUE, nodePointer2.getIndex());
      assertEquals(44, int0);
      assertEquals(Integer.MIN_VALUE, nodePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, (QName) null, (Object) null);
      int int0 = nodePointer0.compareTo(variablePointer0);
      assertEquals(Integer.MIN_VALUE, nodePointer0.getIndex());
      assertFalse(nodePointer0.isAttribute());
      assertEquals(1, int0);
  }
}