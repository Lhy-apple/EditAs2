/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:44:21 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.jdom;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import org.apache.commons.jxpath.BasicVariables;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.ri.NamespaceResolver;
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
import org.jdom.CDATA;
import org.jdom.Comment;
import org.jdom.DocType;
import org.jdom.Document;
import org.jdom.Element;
import org.jdom.ProcessingInstruction;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JDOMNodePointer_ESTest extends JDOMNodePointer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      boolean boolean0 = jDOMNodePointer0.isCollection();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, "F8Q&V4J\"t{");
      NodePointer nodePointer0 = jDOMNodePointer0.namespacePointer("F8Q&V4J\"t{");
      assertFalse(nodePointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Element element0 = new Element("mns", "mns", "mns");
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "mns");
      boolean boolean0 = jDOMNodePointer0.isLanguage("mns");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Locale locale0 = Locale.ITALY;
      BasicVariables basicVariables0 = new BasicVariables();
      QName qName0 = new QName("xml");
      VariablePointer variablePointer0 = new VariablePointer(basicVariables0, qName0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) variablePointer0, locale0);
      int int0 = jDOMNodePointer0.getLength();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Element element0 = new Element("xmlns", "xmlns");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, (String) null);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, (String) null);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertEquals("xmlns", nodeNameTest0.toString());
      assertFalse(boolean0);
      assertEquals("xmlns", jDOMNodePointer0.getNamespaceURI());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Element element0 = new Element("mns", "mns", "mns");
      Locale locale0 = Locale.FRANCE;
      QName qName0 = new QName("mns");
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("mns");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstructionTest0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) nodeTypeTest0);
      // Undeclared exception!
      try { 
        jDOMNodePointer1.createChild(jXPathContext0, qName0, Integer.MIN_VALUE);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Factory is not set on the JXPathContext - cannot create path: /mns:mns[1]
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(jDOMNodePointer0, qName0, qName0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.compareChildNodePointers(nodePointer0, jDOMNodePointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // JXPath internal error: compareChildNodes called for zh_TW
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      QName qName0 = new QName("<<unknown namespace>>", "http://www.w3.org/2000/xmlns/");
      NodeIterator nodeIterator0 = jDOMNodePointer0.attributeIterator(qName0);
      assertEquals(0, nodeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      QName qName0 = new QName((String) null, "");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(variablePointer0, variablePointer0);
      jDOMNodePointer0.hashCode();
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Locale locale0 = Locale.KOREAN;
      Comment comment0 = new Comment("#H}V8Em=S<@");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "#H}V8Em=S<@");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) comment0);
      QName qName0 = new QName("<<unknown namespace>>", "C0,tTPVy>");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild(jXPathContext0, qName0, 256, (Object) "C0,tTPVy>");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Factory is not set on the JXPathContext - cannot create path: id('#H}V8Em=S<@')
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      String string0 = jDOMNodePointer0.getNamespaceURI();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Element element0 = new Element("xmlns");
      Locale locale0 = Locale.ENGLISH;
      QName qName0 = new QName("xmlns", "xmlns");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodeNameTest0, locale0, "xmlns");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      String string0 = jDOMNodePointer1.toString();
      assertEquals("id('xmlns')/xmlns[1]", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      NamespaceResolver namespaceResolver0 = jDOMNodePointer0.getNamespaceResolver();
      NamespaceResolver namespaceResolver1 = jDOMNodePointer0.getNamespaceResolver();
      assertSame(namespaceResolver1, namespaceResolver0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Element element0 = new Element("xmlns", "xmlns");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, (String) null);
      String string0 = jDOMNodePointer0.getNamespaceURI("");
      assertEquals("xmlns", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CDATA cDATA0 = new CDATA("I~y'bKMy)*3F");
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0, "I~y'bKMy)*3F");
      String string0 = jDOMNodePointer0.getNamespaceURI("xml");
      assertEquals("http://www.w3.org/XML/1998/namespace", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Element element0 = new Element("xmlns");
      Locale locale0 = Locale.ITALIAN;
      Document document0 = new Document(element0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0, (String) null);
      String string0 = jDOMNodePointer0.getNamespaceURI("<<unknown namespace>>");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("FhI`");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, "F8Q&V4J\"t{");
      int int0 = jDOMNodePointer0.compareChildNodePointers(jDOMNodePointer0, jDOMNodePointer0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Element element0 = new Element("xmlns");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "http://www.w3.org/2000/xmlns/");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) element0);
      NodePointer nodePointer0 = jDOMNodePointer0.createPath(jXPathContext0, (Object) jXPathContext0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer((Object) nodePointer0, locale0);
      int int0 = jDOMNodePointer0.compareChildNodePointers(nodePointer0, jDOMNodePointer1);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Element element0 = new Element("xmlns", "xmlns");
      Document document0 = new Document(element0);
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Element element0 = new Element("mns", "mns", "mns");
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Element element0 = new Element("ls");
      Locale locale0 = Locale.ROOT;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "http://www.w3.org/XML/1998/namespace");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) "ls");
      jDOMNodePointer0.createPath(jXPathContext0, (Object) locale0);
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
      Element element0 = new Element("mns", "mns", "mns");
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "mns");
      QName qName0 = jDOMNodePointer0.getName();
      assertEquals("mns:mns", qName0.toString());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("org.jdom.IllegalNameException", hashMap0);
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0, "#- 7!8nw5a.");
      QName qName0 = jDOMNodePointer0.getName();
      assertEquals("org.jdom.IllegalNameException", qName0.toString());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Element element0 = new Element("xmlns");
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, (String) null);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) null);
      jDOMNodePointer0.createPath(jXPathContext0, (Object) locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Element element0 = new Element("xmlns", "xmlns");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "<<unknown namespace>>");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) "<<unknown namespace>>");
      Document document0 = new Document(element0, (DocType) null, "<<unknown namespace>>");
      NodePointer nodePointer0 = jDOMNodePointer0.createPath(jXPathContext0, (Object) document0);
      nodePointer0.getValue();
      assertEquals(1, element0.getContentSize());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Element element0 = new Element("xmlns", "xmlns");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "<<unknown namespace>>");
      Comment comment0 = new Comment("http://www.w3.org/2000/xmlns/");
      jDOMNodePointer0.setValue(comment0);
      Object object0 = jDOMNodePointer0.getValue();
      assertNotNull(object0);
      assertEquals("", object0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("org.jdom.IllegalNameException", hashMap0);
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0, (String) null);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("", object0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CDATA cDATA0 = new CDATA("}0Xo#//$>gF]c5_");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, (Locale) null, "}0Xo#//$>gF]c5_");
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
      CDATA cDATA0 = new CDATA((String) null);
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0, (String) null);
      jDOMNodePointer0.setValue("/text()[1]");
      assertEquals("/text()[1]", cDATA0.getValue());
      assertEquals("/text()[1]", cDATA0.getText());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CDATA cDATA0 = new CDATA("jNva.lang.Integer@0000000-03");
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0, "jNva.lang.Integer@0000000-03");
      LinkedList<Locale.LanguageRange> linkedList0 = new LinkedList<Locale.LanguageRange>();
      Locale.FilteringMode locale_FilteringMode0 = Locale.FilteringMode.MAP_EXTENDED_RANGES;
      List<String> list0 = Locale.filterTags((List<Locale.LanguageRange>) linkedList0, (Collection<String>) null, locale_FilteringMode0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.setValue(list0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Element element0 = new Element("xmlns", "xmlns");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, (String) null);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) null);
      NodePointer nodePointer0 = jDOMNodePointer0.createPath(jXPathContext0, (Object) element0);
      assertEquals(1, nodePointer0.getLength());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Element element0 = new Element("xlnXs", "xlnXs", "xlnXs");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "http://www.w3.org/XML/1998/namespace");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) element0);
      CDATA cDATA0 = new CDATA("<<unknown namespace>>");
      NodePointer nodePointer0 = jDOMNodePointer0.createPath(jXPathContext0, (Object) cDATA0);
      assertTrue(nodePointer0.isNode());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Element element0 = new Element("xmlns");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, (String) null);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) null);
      NodePointer nodePointer0 = jDOMNodePointer0.createPath(jXPathContext0, (Object) null);
      assertFalse(nodePointer0.isCollection());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Element element0 = new Element("xmlns");
      Locale locale0 = Locale.GERMANY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, (String) null);
      jDOMNodePointer0.setValue("");
      assertFalse(jDOMNodePointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Element element0 = new Element("mns");
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "mns");
      DocType docType0 = new DocType("qcWf", "<<unknown namespace>>");
      Document document0 = new Document(element0, docType0, "http://www.w3.org/2000/xmlns/");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) docType0);
      jDOMNodePointer0.createPath(jXPathContext0, (Object) document0);
      assertEquals(1, element0.getContentSize());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(3);
      Locale locale0 = Locale.GERMAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      NodeNameTest nodeNameTest0 = new NodeNameTest((QName) null);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Element element0 = new Element("xmlns", "xmlns");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, (String) null);
      QName qName0 = new QName("http://www.w3.org/2000/xmlns/");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, (String) null);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Element element0 = new Element("xmlns");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, (String) null);
      QName qName0 = new QName("<<unknown namespace>>", "xmlns");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, (String) null);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Element element0 = new Element("xmlns");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, (String) null);
      QName qName0 = new QName("<<unknown namespace>>", "xmlns");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "dA");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Element element0 = new Element("xmlns");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, (String) null);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "<<unknown namespace>>");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertEquals("xmlns", nodeNameTest0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("org.jdom.IllegalNameException", hashMap0);
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0, (String) null);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("org.jdom.IllegalNameException");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) processingInstructionTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Locale locale0 = Locale.UK;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0, (String) null);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(2);
      Locale locale0 = Locale.GERMAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("http://www.w3.org/2000/xmlns/", locale0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(4);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Locale locale0 = Locale.GERMAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(2);
      CDATA cDATA0 = new CDATA("594%~KEt+R{%@(I6");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("http://www.w3.org/2000/xmlns/");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      String string0 = JDOMNodePointer.getPrefix((Object) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Element element0 = new Element("mns", "mns", "mns");
      String string0 = JDOMNodePointer.getPrefix(element0);
      assertEquals("mns", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      String string0 = JDOMNodePointer.getLocalName(jDOMNodePointer0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Locale locale0 = Locale.CANADA;
      Comment comment0 = new Comment("#H}V8Em=S<'@");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "#H}V8Em=S<'@");
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
  public void test55()  throws Throwable  {
      Element element0 = new Element("xmlns", "xmlns");
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, (String) null);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) null);
      QName qName0 = new QName("http://www.w3.org/XML/1998/namespace", "http://www.w3.org/2000/xmlns/");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
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
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) null);
      QName qName0 = jDOMNodePointer0.getName();
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
  public void test57()  throws Throwable  {
      Element element0 = new Element("xmlns");
      Locale locale0 = Locale.CHINA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, (String) null);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) null);
      QName qName0 = jDOMNodePointer0.getName();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The name \"xmlns\" is not legal for JDOM/XML attributes: An Attribute name may not be \"xmlns\"; use the Namespace class to manage namespaces.
         //
         verifyException("org.jdom.Attribute", e);
      }
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Element element0 = new Element("xmlns", "xmlns");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, (String) null);
      String string0 = jDOMNodePointer0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      CDATA cDATA0 = new CDATA((String) null);
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0, (String) null);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/text()[1]", string0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("org.jdom.IllegalNameException", hashMap0);
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0, (String) null);
      String string0 = jDOMNodePointer0.toString();
      assertEquals("/processing-instruction('org.jdom.IllegalNameException')[1]", string0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Locale locale0 = Locale.UK;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer((Object) null, locale0, (String) null);
      boolean boolean0 = jDOMNodePointer1.equals(jDOMNodePointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Integer integer0 = new Integer(4476);
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(integer0, locale0);
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Object object0 = new Object();
      Locale locale0 = Locale.GERMANY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(object0, locale0);
      boolean boolean0 = jDOMNodePointer0.equals(locale0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Integer integer0 = new Integer(4476);
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(integer0, locale0);
      Integer integer1 = new Integer(4476);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(integer1, locale0, "z");
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer1);
      assertFalse(boolean0);
  }
}
