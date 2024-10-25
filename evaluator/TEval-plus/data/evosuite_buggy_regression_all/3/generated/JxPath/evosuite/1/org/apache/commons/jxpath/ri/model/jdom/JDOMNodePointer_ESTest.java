/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:56:31 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.jdom;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Locale;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.compiler.NodeNameTest;
import org.apache.commons.jxpath.ri.compiler.NodeTest;
import org.apache.commons.jxpath.ri.compiler.NodeTypeTest;
import org.apache.commons.jxpath.ri.compiler.ProcessingInstructionTest;
import org.apache.commons.jxpath.ri.model.NodeIterator;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.VariablePointer;
import org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jdom.Attribute;
import org.jdom.CDATA;
import org.jdom.Comment;
import org.jdom.Content;
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
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("attribute namespace");
      Locale locale0 = Locale.PRC;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstructionTest0, locale0);
      boolean boolean0 = jDOMNodePointer0.isCollection();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Locale locale0 = Locale.KOREA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0);
      NodePointer nodePointer0 = jDOMNodePointer0.namespacePointer("+]qcxb%{");
      assertFalse(nodePointer0.isCollection());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Comment comment0 = new Comment("org.apahe.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.TAIWAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0);
      int int0 = jDOMNodePointer0.getLength();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Locale locale0 = Locale.ROOT;
      Element element0 = new Element("lang", "lang", "lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
      assertEquals("lang", qName0.getPrefix());
      assertEquals("lang", qName0.getName());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, "Ko.");
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      NodeIterator nodeIterator0 = jDOMNodePointer0.childIterator(nodeTypeTest0, false, (NodePointer) null);
      assertEquals(0, nodeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeIterator nodeIterator0 = jDOMNodePointer0.namespaceIterator();
      assertEquals(0, nodeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Element element0 = new Element("g", "g", "g");
      Locale locale0 = new Locale("g", "g", "g");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      int int0 = jDOMNodePointer0.compareChildNodePointers(jDOMNodePointer0, jDOMNodePointer0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      Element element0 = new Element("lng", "lng", "lng");
      Namespace namespace0 = Namespace.getNamespace("lng");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(namespace0, locale0, "lng");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      // Undeclared exception!
      try { 
        jDOMNodePointer1.asPath();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Locale locale0 = Locale.PRC;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("f*%c!m?*|O", locale0, "f*%c!m?*|O");
      jDOMNodePointer0.hashCode();
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Locale locale0 = Locale.US;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) jDOMNodePointer0);
      QName qName0 = new QName("http://www.w3.org/XML/1998/namespace");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "http://www.w3.org/XML/1998/namespace");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild(jXPathContext0, qName0, (-1556), (Object) nodeNameTest0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Factory is not set on the JXPathContext - cannot create path: 
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, "Ko.");
      String string0 = jDOMNodePointer0.getNamespaceURI();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Locale locale0 = Locale.US;
      Element element0 = new Element("lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "lang");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
      assertEquals("lang", qName0.getName());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("lang", (Locale) null);
      String string0 = jDOMNodePointer0.getNamespaceURI("http://www.w3.org/2000/xmlns/");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Element element0 = new Element("ang", "ang", "ang");
      Document document0 = new Document(element0);
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      Element element0 = new Element("lang", "lang", "lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("org.jdom.Attribute");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Locale locale0 = Locale.US;
      Element element0 = new Element("lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, jDOMNodePointer0);
      // Undeclared exception!
      try { 
        jDOMNodePointer1.compareChildNodePointers(jDOMNodePointer1, jDOMNodePointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // JXPath internal error: compareChildNodes called for 
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Comment comment0 = new Comment("g");
      Element element0 = new Element("g", "g", "g");
      element0.setContent((Content) comment0);
      Locale locale0 = new Locale("g", "g", "g");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NodePointer nodePointer0 = variablePointer0.getImmediateValuePointer();
      int int0 = jDOMNodePointer0.compareChildNodePointers(nodePointer0, jDOMNodePointer0);
      assertEquals("g", qName0.getName());
      assertEquals("g", qName0.getPrefix());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("lang", (Locale) null);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Locale locale0 = Locale.US;
      Element element0 = new Element("lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Locale locale0 = Locale.US;
      Element element0 = new Element("lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue(locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Element element0 = new Element("ang", "ang", "ang");
      Document document0 = new Document(element0);
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Locale locale0 = new Locale("ag", "ag");
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("ag", "ag");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      assertEquals("ag", qName0.getName());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Locale locale0 = new Locale("ag", "ag");
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("ag", "ag");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertNotNull(object0);
      assertEquals("ag", object0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Locale locale0 = Locale.US;
      Element element0 = new Element("lang", "lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("", object0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Comment comment0 = new Comment("org.apahe.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.TAIWAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("org.apahe.commons.jxpath.ri.model.jdom.JDOMNodePointer", object0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Locale locale0 = Locale.TAIWAN;
      CDATA cDATA0 = new CDATA("<<unknown namespace>>");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("<<unknown namespace>>", object0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      CDATA cDATA0 = new CDATA("<<unknown namespace>>");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
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
  public void test29()  throws Throwable  {
      CDATA cDATA0 = new CDATA("lag");
      Locale locale0 = Locale.CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      jDOMNodePointer0.setValue("lag");
      assertEquals("lag", cDATA0.getValue());
      assertEquals("lag", cDATA0.getText());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Text text0 = new Text("w/}");
      Locale locale0 = Locale.TAIWAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(text0, locale0);
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
      Locale locale0 = Locale.ROOT;
      Element element0 = new Element("lang", "lang", "lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue(element0);
      assertEquals(Integer.MIN_VALUE, jDOMNodePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Element element0 = new Element("ang", "ang");
      Locale locale0 = new Locale("", "Qlg");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "5;8Q");
      Document document0 = new Document();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.setValue(document0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Root element not set
         //
         verifyException("org.jdom.Document", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Locale locale0 = Locale.US;
      Element element0 = new Element("lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      CDATA cDATA0 = new CDATA("<<unknown namespace>>");
      jDOMNodePointer0.setValue(cDATA0);
      assertEquals("<<unknown namespace>>", cDATA0.getText());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Locale locale0 = Locale.US;
      Element element0 = new Element("lang", "lang", "lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      Comment comment0 = new Comment("<<unknown namespace>>");
      jDOMNodePointer0.setValue(comment0);
      assertEquals("<<unknown namespace>>", comment0.getValue());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Locale locale0 = Locale.KOREA;
      Element element0 = new Element("lang", "lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue((Object) null);
      assertFalse(jDOMNodePointer0.isCollection());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Locale locale0 = Locale.ROOT;
      Element element0 = new Element("lang", "lang", "lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue("");
      assertTrue(jDOMNodePointer0.isNode());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Locale locale0 = Locale.US;
      Element element0 = new Element("lang", "lang", "lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("lang", (Locale) null);
      Text text0 = new Text("'CtZM9wj6*c*2~");
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(2);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) text0, (NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      QName qName0 = new QName("http://www.w3.org/XML/1998/namespace");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Locale locale0 = Locale.US;
      Element element0 = new Element("lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("lang", "http://www.w3.org/XML/1998/namespace");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "lang");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("http://www.w3.org/XML/1998/namespace");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodeTypeTest0, locale0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(3);
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodeTypeTest0, locale0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("lang", (Locale) null);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(2);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("ag", "ag");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("Rv;Y1bU@zk");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Locale locale0 = Locale.ROOT;
      Element element0 = new Element("lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Locale locale0 = Locale.ROOT;
      Element element0 = new Element("lang", "lang", "lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "i8O:");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertEquals("lang", qName0.getName());
      assertFalse(boolean0);
      assertEquals("lang", qName0.getPrefix());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Locale locale0 = Locale.KOREA;
      Element element0 = new Element("ljng", "ljng");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "ljng");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Locale locale0 = Locale.US;
      String string0 = JDOMNodePointer.getPrefix(locale0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Element element0 = new Element("ang", "ang");
      String string0 = JDOMNodePointer.getPrefix(element0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Element element0 = new Element("ang", "ang", "ang");
      String string0 = JDOMNodePointer.getPrefix(element0);
      assertEquals("ang", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Attribute attribute0 = new Attribute("TkOg", "TkOg");
      String string0 = JDOMNodePointer.getPrefix(attribute0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Attribute attribute0 = new Attribute("TkOg", "TkOg");
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      attribute0.setNamespace(namespace0);
      String string0 = JDOMNodePointer.getPrefix(attribute0);
      assertEquals("xml", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Namespace namespace0 = Namespace.NO_NAMESPACE;
      Attribute attribute0 = new Attribute("ang", "http://www.w3.org/XML/1998/namespace", namespace0);
      String string0 = JDOMNodePointer.getLocalName(attribute0);
      assertEquals("ang", string0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      String string0 = JDOMNodePointer.getLocalName(locale0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      Element element0 = new Element("lang", "lang", "lang");
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      Attribute attribute0 = new Attribute("lang", "lang", 1, namespace0);
      Element element1 = element0.setAttribute(attribute0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element1, locale0);
      boolean boolean0 = jDOMNodePointer0.isLanguage(":oud~2kn!");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLanguage("http://www.w3.org/XML/1998/namespace");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Locale locale0 = Locale.US;
      Element element0 = new Element("lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.getLanguage();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Locale locale0 = new Locale("ag", "ag");
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("ag", "ag");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
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
  public void test61()  throws Throwable  {
      Comment comment0 = new Comment("g");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, (Locale) null);
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
  public void test62()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, "K`OBI2u8x*g(C$<7`!^");
      QName qName0 = jDOMNodePointer0.getName();
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) jDOMNodePointer0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild(jXPathContext0, qName0, Integer.MIN_VALUE);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Factory is not set on the JXPathContext - cannot create path: id('K`OBI2u8x*g(C$<7`!^')
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      QName qName0 = new QName(":$\"P8%9 &", "<<unknown namespace>>");
      Comment comment0 = new Comment("<<unknown namespace>>");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) comment0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, ":$\"P8%9 &");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot create an attribute for path id(':$&quot;P8%9 &')/@:$\"P8%9 &:<<unknown namespace>>, operation is not allowed for this type of node
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Locale locale0 = Locale.US;
      Element element0 = new Element("lang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
      NodePointer nodePointer0 = jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
      assertEquals(Integer.MIN_VALUE, nodePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Element element0 = new Element("laDng", "laDng");
      Locale locale0 = Locale.KOREAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) locale0);
      QName qName0 = new QName("laDng", "<<unknown namespace>>");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Unknown namespace prefix: laDng
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Locale locale0 = Locale.US;
      Element element0 = new Element("ang", "ang", "ang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("ang", "Psel");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) Integer.MIN_VALUE);
      jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      NodePointer nodePointer0 = jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      assertFalse(nodePointer0.isRoot());
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Comment comment0 = new Comment("g");
      Element element0 = new Element("g", "g", "g");
      element0.setContent((Content) comment0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, (Locale) null);
      jDOMNodePointer0.remove();
      assertEquals(1, jDOMNodePointer0.getLength());
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Element element0 = new Element("ang", "ang", "ang");
      Locale locale0 = new Locale("ang", "ang", "ang");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Comment comment0 = new Comment("ljng");
      Locale locale0 = Locale.KOREA;
      Element element0 = new Element("ljng", "The XML construct ");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "ljng");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer((Object) jDOMNodePointer0, locale0);
      JDOMNodePointer jDOMNodePointer2 = new JDOMNodePointer(jDOMNodePointer1, element0);
      // Undeclared exception!
      try { 
        jDOMNodePointer2.asPath();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      CDATA cDATA0 = new CDATA((String) null);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/text()[1]", string0);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("ag", "ag");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/processing-instruction('ag')[1]", string0);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("')", locale0, "')");
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("id('&apos;)')", string0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("org.apache.comons.jxpath.ri.QName@0000000001");
      Locale locale0 = Locale.CHINA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstructionTest0, locale0, "laDng");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(processingInstructionTest0, locale0);
      boolean boolean0 = jDOMNodePointer1.equals(jDOMNodePointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, "K`OBI2u8x*g(C$<7`!^");
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.equals("http://www.w3.org/XML/1998/namespace");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, "K`OBI2u8x*g(C$<7`!^");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer("K`OBI2u8x*g(C$<7`!^", locale0);
      boolean boolean0 = jDOMNodePointer1.equals(jDOMNodePointer0);
      assertFalse(boolean0);
  }
}
