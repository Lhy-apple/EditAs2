/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:05:13 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.jdom;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.Locale;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.ri.NamespaceResolver;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.compiler.NodeNameTest;
import org.apache.commons.jxpath.ri.compiler.NodeTest;
import org.apache.commons.jxpath.ri.compiler.NodeTypeTest;
import org.apache.commons.jxpath.ri.compiler.ProcessingInstructionTest;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.VariablePointer;
import org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jdom.Attribute;
import org.jdom.CDATA;
import org.jdom.Comment;
import org.jdom.DocType;
import org.jdom.Document;
import org.jdom.Element;
import org.jdom.Namespace;
import org.jdom.ProcessingInstruction;
import org.jdom.Text;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JDOMNodePointer_ESTest extends JDOMNodePointer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.isCollection();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Locale locale0 = Locale.GERMANY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodePointer nodePointer0 = jDOMNodePointer0.namespacePointer((String) null);
      assertEquals(1, nodePointer0.getLength());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Element element0 = new Element("contains", "contains", "contains");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.isLanguage("contains");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Comment comment0 = new Comment("<>;x:*{tP");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, (Locale) null, "<>;x:*{tP");
      int int0 = jDOMNodePointer0.getLength();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("BO", "BO");
      Locale locale0 = new Locale("BO", "BO", "BO");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("http://www.w3.org/2000/xmlns/");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      int int0 = jDOMNodePointer0.compareChildNodePointers(jDOMNodePointer0, jDOMNodePointer0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Element element0 = new Element("coa");
      Locale locale0 = Locale.forLanguageTag("coa");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, "coa");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      String string0 = jDOMNodePointer1.asPath();
      assertEquals("id('coa')/coa[1]", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Locale locale0 = Locale.UK;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(jDOMNodePointer0, qName0, "http://www.w3.org/2000/xmlns/");
      int int0 = nodePointer0.compareTo(jDOMNodePointer0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      JXPathContext jXPathContext0 = JXPathContext.newContext((JXPathContext) null, (Object) jDOMNodePointer0);
      QName qName0 = jDOMNodePointer0.getName();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild(jXPathContext0, qName0, (-4794), (Object) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Factory is not set on the JXPathContext - cannot create path: 
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Locale locale0 = Locale.TAIWAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Element element0 = new Element("contains", "contains", "contains");
      QName qName0 = new QName("contains");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) variablePointer0, (Object) element0, (NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Element element0 = new Element("coa", "coa", "coa");
      Locale locale0 = Locale.forLanguageTag("coa");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, "coa");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      NamespaceResolver namespaceResolver0 = jDOMNodePointer1.getNamespaceResolver();
      assertNotNull(namespaceResolver0);
      
      String string0 = jDOMNodePointer1.asPath();
      assertEquals("id('coa')/coa:coa[1]", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Locale locale0 = Locale.UK;
      Text text0 = new Text("B<6w^");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(text0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("xml");
      assertEquals("http://www.w3.org/XML/1998/namespace", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document();
      Element element0 = new Element("conans");
      Document document1 = document0.setRootElement(element0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document1, (Locale) null);
      String string0 = jDOMNodePointer0.getNamespaceURI("L=7#,k.31*CPTy'PhE");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Element element0 = new Element("contains", "contains", "contains");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, (Locale) null, "contains");
      QName qName0 = jDOMNodePointer0.getName();
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) jDOMNodePointer0);
      jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      assertEquals("contains:contains", qName0.toString());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer((Object) null, locale0, "<<unknown namespace>>");
      // Undeclared exception!
      try { 
        jDOMNodePointer1.compareChildNodePointers(jDOMNodePointer0, jDOMNodePointer1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // JXPath internal error: compareChildNodes called for null
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Element element0 = new Element("conta", "conta");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, "conta");
      element0.setText("http://www.w3.org/2000/xmlns/");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      int int0 = jDOMNodePointer1.compareChildNodePointers(jDOMNodePointer1, jDOMNodePointer0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Element element0 = new Element("org.apace.commons.jxpath.JXPathAstractFactoryExcetion", "org.apace.commons.jxpath.JXPathAstractFactoryExcetion");
      Document document0 = new Document(element0);
      Locale locale0 = new Locale("org.apace.commons.jxpath.JXPathAstractFactoryExcetion", "org.apace.commons.jxpath.JXPathAstractFactoryExcetion", "org.apace.commons.jxpath.JXPathAstractFactoryExcetion");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Element element0 = new Element("org.apache.comLons.jxpath.JXPathContextFactory", "org.apache.comLons.jxpath.JXPathContextFactory");
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Element element0 = new Element("org.apache.comLons.jxpath.JXPathContextFactory", "org.apache.comLons.jxpath.JXPathContextFactory");
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      element0.setText("http://www.w3.org/XML/1998/namespace");
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Locale locale0 = Locale.TAIWAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      Element element0 = new Element("contains");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "contains");
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "http://www.w3.org/XML/1998/namespace");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) element0, (NodeTest) nodeNameTest0);
      assertTrue(boolean0);
      assertEquals("contains", nodeNameTest0.toString());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("BO", "BO");
      Locale locale0 = new Locale("BO", "BO", "BO");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      assertEquals("BO", qName0.toString());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Element element0 = new Element("org.apache.comLons.jxpath.JXPathContextFactory", "org.apache.comLons.jxpath.JXPathContextFactory");
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      element0.setText("org.apache.comLons.jxpath.JXPathContextFactory");
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("org.apache.comLons.jxpath.JXPathContextFactory", object0);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Element element0 = new Element("OTXQx", "OTXQx", "OTXQx");
      Locale locale0 = Locale.UK;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      Document document0 = new Document(element0);
      jDOMNodePointer0.setValue(document0);
      jDOMNodePointer0.getValue();
      assertEquals(1, element0.getContentSize());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.JXPathContextFactory", "org.apache.commons.jxpath.JXPathContextFactory");
      Locale locale0 = Locale.KOREAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("org.apace.commons.jxpath.JXPathAstractFactoryExcetion", "org.apache.commons.jxpath.JXPathContextFactory");
      jDOMNodePointer0.setValue(processingInstruction0);
      Object object0 = jDOMNodePointer0.getValue();
      assertNotNull(object0);
      assertEquals("", object0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Comment comment0 = new Comment("')");
      Locale locale0 = Locale.US;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "')");
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("')", object0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("BO", "BO");
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("BO", object0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("')", locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CDATA cDATA0 = new CDATA("coa");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, cDATA0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.setValue((Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CDATA cDATA0 = new CDATA("");
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.setValue("");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CDATA cDATA0 = new CDATA("Factory could not create a child node for path: ");
      Locale locale0 = new Locale("Factory could not create a child node for path: ");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      jDOMNodePointer0.setValue("Factory could not create a child node for path: ");
      assertEquals("Factory could not create a child node for path: ", cDATA0.getValue());
      assertEquals("Factory could not create a child node for path: ", cDATA0.getText());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Element element0 = new Element("contains", "contains");
      Locale locale0 = Locale.GERMAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue(element0);
      assertTrue(jDOMNodePointer0.isNode());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Element element0 = new Element("contains", "contains", "contains");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      CDATA cDATA0 = new CDATA("<<unknown namespace>>");
      jDOMNodePointer0.setValue(cDATA0);
      assertFalse(jDOMNodePointer0.isContainer());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Locale locale0 = Locale.CANADA;
      Element element0 = new Element("org.apah.common.jxpath.ClassFunc8ions");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      Comment comment0 = new Comment("OYXQx");
      jDOMNodePointer0.setValue(comment0);
      assertFalse(jDOMNodePointer0.isContainer());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Element element0 = new Element("contains", "contains", "contains");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      jDOMNodePointer0.setValue((Object) null);
      assertEquals(Integer.MIN_VALUE, jDOMNodePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Element element0 = new Element("contains", "contains");
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue(locale0);
      assertTrue(jDOMNodePointer0.isNode());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Element element0 = new Element("contains", "contains");
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      jDOMNodePointer0.setValue(linkedList0);
      assertTrue(jDOMNodePointer0.isNode());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      Element element0 = new Element("org.apah.common.jxpath.ClassFunc8ions", "org.apah.common.jxpath.ClassFunc8ions");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      DocType docType0 = new DocType("lang", "<<unknown namespace>>");
      Document document0 = new Document(element0, docType0);
      jDOMNodePointer0.setValue(document0);
      assertEquals(1, element0.getContentSize());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      QName qName0 = new QName("\"");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) null, (NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Element element0 = new Element("contains", "contains");
      QName qName0 = new QName("http://www.w3.org/2000/xmlns/");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) variablePointer0, (Object) element0, (NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Element element0 = new Element("contains");
      QName qName0 = new QName("contains");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) variablePointer0, (Object) element0, (NodeTest) nodeNameTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Element element0 = new Element("contains");
      QName qName0 = new QName("contains", "contains");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "contains");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) variablePointer0, (Object) element0, (NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(3);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) null, (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) null, (NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(2);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) null, (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(4);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) "//www.w3.org/2000/xmlns/", (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Locale locale0 = Locale.TAIWAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("http://www.w3.org/XML/1998/namespace");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) processingInstructionTest0, (NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Attribute attribute0 = new Attribute("ctains", "ctains");
      String string0 = JDOMNodePointer.getPrefix(attribute0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Element element0 = new Element("contains", "contains", "contains");
      String string0 = JDOMNodePointer.getPrefix(element0);
      assertNotNull(string0);
      assertEquals("contains", string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = new Document();
      String string0 = JDOMNodePointer.getPrefix(document0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      Attribute attribute0 = new Attribute("contains", "contains", namespace0);
      String string0 = JDOMNodePointer.getPrefix(attribute0);
      assertEquals("xml", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Comment comment0 = new Comment("<>;x:*{tP");
      String string0 = JDOMNodePointer.getLocalName(comment0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      Attribute attribute0 = new Attribute("contains", "contains", namespace0);
      String string0 = JDOMNodePointer.getLocalName(attribute0);
      assertEquals("contains", string0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Comment comment0 = new Comment("#.H;,");
      Locale locale0 = Locale.US;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "#.H;,");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.remove();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot remove root JDOM node
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) jDOMNodePointer0);
      QName qName0 = new QName("<<unknown namespace>>");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild(jXPathContext0, qName0, Integer.MIN_VALUE);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Factory is not set on the JXPathContext - cannot create path: 
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      QName qName0 = jDOMNodePointer0.getName();
      Object object0 = new Object();
      JXPathContext jXPathContext0 = JXPathContext.newContext(object0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot create an attribute for path /@null, operation is not allowed for this type of node
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Element element0 = new Element("contains", "contains", "contains");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, (Locale) null, "contains");
      QName qName0 = new QName("<<unknown namespace>>", "http://www.w3.org/2000/xmlns/");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) "contains");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Unknown namespace prefix: <<unknown namespace>>
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Document document0 = new Document();
      Element element0 = new Element("conans", (String) null);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, (Locale) null);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) document0);
      QName qName0 = jDOMNodePointer0.getName();
      jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      assertEquals("conans", qName0.toString());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Element element0 = new Element("contains", "contains");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Element element0 = new Element("conta", "conta", "conta");
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("conta");
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstructionTest0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      String string0 = jDOMNodePointer1.asPath();
      assertEquals("/conta:conta[1]", string0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Locale locale0 = Locale.KOREA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      Text text0 = new Text("http://www.w3.org/XML/1998/namespace");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, text0);
      String string0 = jDOMNodePointer1.asPath();
      assertEquals("/text()[1]", string0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("BO", "BO");
      Locale locale0 = new Locale("BO", "BO", "BO");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/processing-instruction('BO')[1]", string0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Comment comment0 = new Comment("')");
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("')", locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(comment0, locale0);
      boolean boolean0 = jDOMNodePointer1.equals(jDOMNodePointer0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Locale locale0 = Locale.KOREAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Locale locale0 = Locale.KOREAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.equals("org.apache.commons.jxpath.ri.compiler.NodeTypeTest@0000000019");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("')", locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(locale0, locale0, "<<unknown namespace>>");
      JDOMNodePointer jDOMNodePointer2 = new JDOMNodePointer((NodePointer) jDOMNodePointer0, (Object) locale0);
      boolean boolean0 = jDOMNodePointer1.equals(jDOMNodePointer2);
      assertTrue(boolean0);
      assertFalse(jDOMNodePointer2.equals((Object)jDOMNodePointer0));
  }
}
