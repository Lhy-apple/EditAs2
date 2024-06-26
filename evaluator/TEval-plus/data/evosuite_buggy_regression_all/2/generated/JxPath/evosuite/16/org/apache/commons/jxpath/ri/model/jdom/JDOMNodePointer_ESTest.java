/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:18:02 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.jdom;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Locale;
import org.apache.commons.jxpath.BasicVariables;
import org.apache.commons.jxpath.JXPathContext;
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
import org.jdom.Content;
import org.jdom.Document;
import org.jdom.Element;
import org.jdom.ProcessingInstruction;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JDOMNodePointer_ESTest extends JDOMNodePointer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Locale locale0 = Locale.TAIWAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.isCollection();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CDATA cDATA0 = new CDATA("Pg*7/*'`&uU%SS");
      Locale locale0 = Locale.CHINA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      NodePointer nodePointer0 = jDOMNodePointer0.namespacePointer("<<unknown namespace>>");
      assertFalse(nodePointer0.isCollection());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Locale locale0 = Locale.ITALY;
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodeTypeTest0, locale0, "http://www.w3.org/2000/xmlns/");
      boolean boolean0 = jDOMNodePointer0.isLanguage("java.lang.Object@0000000004");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Comment comment0 = new Comment("*DnM~vPn");
      Locale locale0 = Locale.KOREA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "*DnM~vPn");
      int int0 = jDOMNodePointer0.getLength();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer");
      Locale locale0 = Locale.forLanguageTag("org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("http://www.w3.org/2000/xmlns/", "org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "http://www.w3.org/2000/xmlns/");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest((-4777));
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodeTypeTest0, (Locale) null);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      assertEquals("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", jDOMNodePointer1.getNamespaceURI());
      
      String string0 = jDOMNodePointer1.toString();
      assertEquals("/org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer:org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer[1]", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CDATA cDATA0 = new CDATA("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.ITALY;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, cDATA0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodeTypeTest0, locale0, "http://www.w3.org/2000/xmlns/");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.compareChildNodePointers(nodePointer0, jDOMNodePointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // JXPath internal error: compareChildNodes called for UNKNOWN()
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      assertEquals("org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer", qName0.toString());
      
      NodePointer nodePointer0 = jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(34);
      boolean boolean0 = JDOMNodePointer.testNode(nodePointer0, (Object) element0, (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      HashMap<JDOMNodePointer, CDATA> hashMap0 = new HashMap<JDOMNodePointer, CDATA>();
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", locale0);
      CDATA cDATA0 = new CDATA("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      CDATA cDATA1 = hashMap0.put(jDOMNodePointer0, cDATA0);
      assertNull(cDATA1);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) locale0);
      QName qName0 = jDOMNodePointer0.getName();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild(jXPathContext0, qName0, 1475, (Object) locale0);
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
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer");
      Locale locale0 = new Locale("org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
      assertEquals("org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer:org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer", qName0.toString());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("xml", "<<unknown namespace>>");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The name \"<<unknown namespace>>\" is not legal for JDOM/XML attributes: XML names cannot begin with the character \"<\".
         //
         verifyException("org.jdom.Attribute", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      LinkedList<ProcessingInstructionTest> linkedList0 = new LinkedList<ProcessingInstructionTest>();
      Document document0 = new Document(linkedList0);
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0, "W97@`3;BU lsG");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.getNamespaceURI("!/Ao(xP|>6u,");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Root element not set
         //
         verifyException("org.jdom.Document", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("<<unknown namespace>>");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("", "xml");
      Object object0 = new Object();
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, object0, locale0);
      int int0 = jDOMNodePointer0.compareChildNodePointers(nodePointer0, nodePointer0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.GERMANY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) jDOMNodePointer0);
      NodePointer nodePointer0 = jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NodePointer nodePointer1 = variablePointer0.getImmediateValuePointer();
      int int0 = jDOMNodePointer0.compareChildNodePointers(nodePointer0, nodePointer1);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      CDATA cDATA0 = new CDATA("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.ITALY;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, cDATA0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodeTypeTest0, locale0, "http://www.w3.org/2000/xmlns/");
      element0.addContent((Content) cDATA0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(nodePointer0, element0);
      int int0 = jDOMNodePointer1.compareChildNodePointers(nodePointer0, jDOMNodePointer0);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      CDATA cDATA0 = new CDATA("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.ITALY;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, cDATA0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodeTypeTest0, locale0, "http://www.w3.org/2000/xmlns/");
      QName qName0 = new QName("http://www.w3.org/XML/1998/namespace");
      element0.addContent((Content) cDATA0);
      NodePointer nodePointer1 = NodePointer.newChildNodePointer(nodePointer0, qName0, nodePointer0.WHOLE_COLLECTION);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(nodePointer1, element0);
      int int0 = jDOMNodePointer1.compareChildNodePointers(nodePointer1, jDOMNodePointer0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document();
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0, "g$da7zi(`VCTy?oQbLB");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.isLeaf();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Root element not set
         //
         verifyException("org.jdom.Document", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer");
      Locale locale0 = Locale.forLanguageTag("org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.modl.jdom.JDOModePointer");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue("org.apache.commons.jxpath.ri.modl.jdom.JDOModePointer");
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, (Locale) null);
      QName qName0 = jDOMNodePointer0.getName();
      assertNull(qName0.getPrefix());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      Document document0 = new Document(element0);
      jDOMNodePointer0.setValue(document0);
      jDOMNodePointer0.getValue();
      assertEquals(1, element0.getContentSize());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.GERMANY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      Comment comment0 = new Comment("http://www.w3.org/XML/1998/namespace");
      jDOMNodePointer0.setValue(comment0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("", object0);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.GERMANY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Object object0 = jDOMNodePointer0.getValue();
      assertNotNull(object0);
      assertEquals("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", object0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Comment comment0 = new Comment("org.apache.commons.jxpath.ri.EvalContext");
      Locale locale0 = new Locale("org.apache.commons.jxpath.ri.EvalContext", "org.apache.commons.jxpath.ri.EvalContext", "org.apache.commons.jxpath.ri.EvalContext");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("org.apache.commons.jxpath.ri.EvalContext", object0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("BYx", "BYx");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, (Locale) null);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("BYx", object0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Locale locale0 = Locale.ITALY;
      CDATA cDATA0 = new CDATA("\"39J>,");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) locale0);
      LinkedList<Locale.LanguageRange> linkedList0 = new LinkedList<Locale.LanguageRange>();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createPath(jXPathContext0, (Object) linkedList0);
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
      CDATA cDATA0 = new CDATA((String) null);
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0, (String) null);
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
  public void test31()  throws Throwable  {
      Locale locale0 = Locale.GERMAN;
      CDATA cDATA0 = new CDATA("&quot;");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      jDOMNodePointer0.setValue(locale0);
      assertEquals("", locale0.getVariant());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.GERMANY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue(element0);
      assertEquals("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", element0.getNamespaceURI());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jpath.ri.model.jdomJDOMNodePointer");
      Locale locale0 = Locale.GERMANY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      CDATA cDATA0 = new CDATA("http://www.w3.org/XML/1998/namespace");
      jDOMNodePointer0.setValue(cDATA0);
      assertTrue(jDOMNodePointer0.isRoot());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(locale0, locale0);
      jDOMNodePointer0.setValue(jDOMNodePointer1);
      assertEquals(1, jDOMNodePointer0.getLength());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue("");
      assertEquals(1, jDOMNodePointer0.getLength());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Element element0 = new Element("org.ache.commons.jxpath.ri.model.domJDOMNodePonter");
      Locale locale0 = Locale.US;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) element0, (NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      QName qName0 = new QName("&quot;");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "&quot;");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) null, (NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOModePointer");
      Locale locale0 = new Locale("org.apache.commons.jxpath.ri.model.jdom.JDOModePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOModePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOModePointer");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("http://www.w3.org/2000/xmlns/");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) element0, (NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) element0, (NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxprth.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "http://www.w3.org/2000/xmlns/");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) element0, (NodeTest) nodeNameTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("Bx", "Bx");
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("d3ombhBV");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) processingInstruction0, (NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(2);
      CDATA cDATA0 = new CDATA("L-E\"T]A*nO4");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) cDATA0, (NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(3);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) nodeTypeTest0, (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(4);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) jDOMNodePointer0, (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Document document0 = new Document();
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      BasicVariables basicVariables0 = new BasicVariables();
      QName qName0 = new QName((String) null, "xml");
      VariablePointer variablePointer0 = new VariablePointer(basicVariables0, qName0);
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, (QName) null, (Object) null);
      boolean boolean0 = JDOMNodePointer.testNode(nodePointer0, (Object) document0, (NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) nodeTypeTest0, (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(2);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) nodeTypeTest0, (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("http://www.w3.org/XML/1998/namespace");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      String string0 = JDOMNodePointer.getPrefix(nodeNameTest0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      String string0 = JDOMNodePointer.getPrefix(element0);
      assertEquals("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Attribute attribute0 = new Attribute("http", "org.apache.commons.jxpath.ri.model.jdom.JDOModePointer", 1);
      String string0 = JDOMNodePointer.getLocalName(attribute0);
      assertEquals("http", string0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      String string0 = JDOMNodePointer.getLocalName("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Comment comment0 = new Comment("org.apache.commons.jxpath.ri.EvalConteht");
      Locale locale0 = new Locale("org.apache.commons.jxpath.ri.EvalConteht", "org.apache.commons.jxpath.ri.EvalConteht", "org.apache.commons.jxpath.ri.EvalConteht");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0);
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
  public void test54()  throws Throwable  {
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      QName qName0 = new QName("http://www.w3.org/XML/1998/namespace", "http://www.w3.org/2000/xmlns/");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot create an attribute for path /@http://www.w3.org/XML/1998/namespace:http://www.w3.org/2000/xmlns/, operation is not allowed for this type of node
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("http://www.w3.org/XML/1998/namespace", (String) null);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Unknown namespace prefix: http://www.w3.org/XML/1998/namespace
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) qName0);
      jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      NodePointer nodePointer0 = jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      assertEquals(Integer.MIN_VALUE, nodePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      CDATA cDATA0 = new CDATA("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.ITALY;
      element0.addContent((Content) cDATA0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0, "http://www.w3.org/2000/xmlns/");
      jDOMNodePointer0.remove();
      assertFalse(jDOMNodePointer0.isCollection());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdomJDOMNodePointer");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest((-4777));
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodeTypeTest0, (Locale) null);
      QName qName0 = new QName("http://www.w3.org/2000/xmlns/");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) null);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      // Undeclared exception!
      try { 
        jDOMNodePointer1.createChild(jXPathContext0, qName0, Integer.MIN_VALUE);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Factory is not set on the JXPathContext - cannot create path: /org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer[1]
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("Bx", "Bx");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, (Locale) null);
      String string0 = jDOMNodePointer0.toString();
      assertEquals("/processing-instruction('Bx')[1]", string0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0, "'^4HL\"!f$U8]x<Q.");
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("id('&apos;^4HL&quot;!f$U8]x<Q.')", string0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer((NodePointer) jDOMNodePointer0, (Object) locale0);
      boolean boolean0 = jDOMNodePointer1.equals(jDOMNodePointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Object object0 = new Object();
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(object0, locale0);
      boolean boolean0 = jDOMNodePointer0.equals(locale0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer((Object) jDOMNodePointer0, locale0);
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer1);
      assertFalse(boolean0);
  }
}
