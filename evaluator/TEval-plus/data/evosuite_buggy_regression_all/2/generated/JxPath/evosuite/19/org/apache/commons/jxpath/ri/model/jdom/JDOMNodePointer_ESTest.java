/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:18:02 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.jdom;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.Locale;
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
      Locale locale0 = Locale.TAIWAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.isCollection();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Locale locale0 = Locale.GERMANY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodePointer nodePointer0 = jDOMNodePointer0.namespacePointer("org.apache.commons.jxpath.JXPathException");
      assertEquals(Integer.MIN_VALUE, nodePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      Element element0 = new Element("preserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLanguage("http://www.w3.org/2000/xmlns/");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Object object0 = new Object();
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(object0, locale0, "id('");
      int int0 = jDOMNodePointer0.getLength();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(18);
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      Element element0 = new Element("preserve", "preserve", "preserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      assertEquals("preserve", jDOMNodePointer1.getNamespaceURI());
      
      String string0 = jDOMNodePointer1.asPath();
      assertEquals("/preserve:preserve[1]", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Locale locale0 = Locale.CHINA;
      Element element0 = new Element("preserve", "preserve", "preserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) "");
      jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      assertEquals("preserve:preserve", qName0.toString());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Namespace namespace0 = Namespace.NO_NAMESPACE;
      Locale locale0 = Locale.TAIWAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(namespace0, locale0);
      jDOMNodePointer0.hashCode();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Comment comment0 = new Comment("Zhw%&#,aJdfxg");
      Locale locale0 = Locale.GERMAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "Zhw%&#,aJdfxg");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) locale0);
      QName qName0 = jDOMNodePointer0.getName();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild(jXPathContext0, qName0, (-2013265917), (Object) "Zhw%&#,aJdfxg");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Factory is not set on the JXPathContext - cannot create path: id('Zhw%&#,aJdfxg')
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      Element element0 = new Element("peUserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "peUserve");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) element0, (NodeTest) nodeNameTest0);
      assertEquals("peUserve", nodeNameTest0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Comment comment0 = new Comment("xml");
      Locale locale0 = Locale.GERMAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "xml");
      String string0 = jDOMNodePointer0.getNamespaceURI("xml");
      assertEquals("http://www.w3.org/XML/1998/namespace", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, (Object) null, locale0);
      LinkedList<NodeNameTest> linkedList0 = new LinkedList<NodeNameTest>();
      Document document0 = new Document(linkedList0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(nodePointer0, document0);
      // Undeclared exception!
      try { 
        jDOMNodePointer1.getNamespaceURI("QVc3LgSa<=");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Root element not set
         //
         verifyException("org.jdom.Document", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("<<unknown namespace>>");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      Element element0 = new Element("preserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("<<unknown namespace>>", "http://www.w3.org/2000/xmlns/");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Unknown namespace prefix: <<unknown namespace>>
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Locale locale0 = Locale.PRC;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("/vYFI>>[D8Gi", locale0, "/vYFI>>[D8Gi");
      int int0 = jDOMNodePointer0.compareChildNodePointers(jDOMNodePointer0, jDOMNodePointer0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("8#.HA\"ZTQSGz31pu?O]", locale0);
      QName qName0 = new QName("<<unknown namespace>>", "http://www.w3.org/XML/1998/namespace");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(jDOMNodePointer0, qName0, jDOMNodePointer0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.compareChildNodePointers(nodePointer0, jDOMNodePointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // JXPath internal error: compareChildNodes called for 8#.HA\"ZTQSGz31pu?O]
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      Element element0 = new Element("p7G.rerve", "p7G.rerve", "p7G.rerve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("p7G.rerve");
      jDOMNodePointer0.setValue(processingInstructionTest0);
      QName qName0 = new QName("http://www.w3.org/XML/1998/namespace");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NodePointer nodePointer0 = variablePointer0.getImmediateValuePointer();
      int int0 = jDOMNodePointer0.compareChildNodePointers(jDOMNodePointer0, nodePointer0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      LinkedList<CDATA> linkedList0 = new LinkedList<CDATA>();
      Document document0 = new Document(linkedList0);
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0);
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
  public void test19()  throws Throwable  {
      Element element0 = new Element("preserve", "preserve", "preserve");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "preserve");
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Element element0 = new Element("preserve", "preserve", "preserve");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "preserve");
      jDOMNodePointer0.setValue("preserve");
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Locale locale0 = Locale.TAIWAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("peserve", "peserve");
      Locale locale0 = Locale.TAIWAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      assertEquals("peserve", qName0.toString());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Element element0 = new Element("preserve", "preserve", "preserve");
      Locale locale0 = Locale.PRC;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "preserve");
      jDOMNodePointer0.setValue(locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Locale locale0 = Locale.GERMANY;
      Element element0 = new Element("peUserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      DocType docType0 = new DocType("peUserve", "http://www.w3.org/XML/1998/namespace", "http://www.w3.org/2000/xmlns/");
      Document document0 = new Document(element0, docType0, "http://www.w3.org/XML/1998/namespace");
      jDOMNodePointer0.setValue(document0);
      jDOMNodePointer0.getValue();
      assertEquals(1, element0.getContentSize());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      Element element0 = new Element("preserve", "preserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      Comment comment0 = new Comment("http://www.w3.org/XML/1998/namespace");
      jDOMNodePointer0.setValue(comment0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("", object0);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Comment comment0 = new Comment("xml");
      Locale locale0 = Locale.GERMAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "xml");
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("xml", object0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("peserve", "peserve");
      Locale locale0 = Locale.UK;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("peserve", object0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CDATA cDATA0 = new CDATA((String) null);
      Locale locale0 = Locale.PRC;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0, (String) null);
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
  public void test29()  throws Throwable  {
      CDATA cDATA0 = new CDATA((String) null);
      Locale locale0 = Locale.TAIWAN;
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
  public void test30()  throws Throwable  {
      CDATA cDATA0 = new CDATA((String) null);
      Locale locale0 = Locale.PRC;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0, (String) null);
      jDOMNodePointer0.setValue(cDATA0);
      assertTrue(jDOMNodePointer0.isNode());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Locale locale0 = Locale.CHINA;
      Element element0 = new Element("p7G.rerve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      Text text0 = new Text("fAcas&ugc!}*");
      jDOMNodePointer0.setValue(text0);
      assertEquals(Integer.MIN_VALUE, jDOMNodePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("peserve", "peserve");
      Element element0 = new Element("peserve");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue(processingInstruction0);
      assertTrue(jDOMNodePointer0.isNode());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      Element element0 = new Element("preserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer((NodePointer) jDOMNodePointer0, (Object) locale0);
      jDOMNodePointer0.setValue(jDOMNodePointer1);
      assertFalse(jDOMNodePointer1.isRoot());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      Element element0 = new Element("preserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue("");
      assertEquals(Integer.MIN_VALUE, NodePointer.WHOLE_COLLECTION);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Element element0 = new Element("presere");
      Locale locale0 = Locale.GERMANY;
      Element element1 = new Element("presere");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element1, locale0);
      Element element2 = element0.addContent("presere");
      jDOMNodePointer0.setValue(element2);
      assertEquals(1, element1.getContentSize());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      QName qName0 = new QName("org.jdom.CDATA@0000000006", "org.jdom.CDATA@0000000006");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) null, (NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Locale locale0 = Locale.TAIWAN;
      Element element0 = new Element("p7.rserve", "p7.rserve", "p7.rserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("http://www.w3.org/2000/xmlns/", "http://www.w3.org/2000/xmlns/");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "http://www.w3.org/2000/xmlns/");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) element0, (NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Element element0 = new Element("peservV");
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) element0, (NodeTest) nodeNameTest0);
      assertTrue(boolean0);
      assertEquals("peservV", nodeNameTest0.toString());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Locale locale0 = Locale.TAIWAN;
      Element element0 = new Element("p7.rserve", "p7.rserve", "p7.rserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) element0, (NodeTest) nodeNameTest0);
      assertFalse(boolean0);
      assertEquals("p7.rserve:p7.rserve", nodeNameTest0.toString());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("peserve", "peserve");
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("peserve", locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("<<unknown namespace>>");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) processingInstruction0, (NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) null, (NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(2);
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodeTypeTest0, locale0);
      CDATA cDATA0 = new CDATA("<<unknown namespace>>");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) cDATA0, (NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(3);
      Locale locale0 = Locale.ENGLISH;
      Object object0 = new Object();
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(object0, locale0);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) jDOMNodePointer0, (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(4);
      QName qName0 = new QName("PSiIB6L-");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) qName0, (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(2);
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) nodeTypeTest0, (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) null, (NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Attribute attribute0 = new Attribute("preserv", "preserv");
      String string0 = JDOMNodePointer.getPrefix(attribute0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Element element0 = new Element("preserve", "preserve", "preserve");
      String string0 = JDOMNodePointer.getPrefix(element0);
      assertEquals("preserve", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      String string0 = JDOMNodePointer.getPrefix("preserve");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      Attribute attribute0 = new Attribute("preserv", "preserv", namespace0);
      String string0 = JDOMNodePointer.getPrefix(attribute0);
      assertEquals("xml", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      Attribute attribute0 = new Attribute("preserv", "preserv", namespace0);
      String string0 = JDOMNodePointer.getLocalName(attribute0);
      assertEquals("preserv", string0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Object object0 = new Object();
      String string0 = JDOMNodePointer.getLocalName(object0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Comment comment0 = new Comment("Zhw%&#,aJdfxg");
      Locale locale0 = Locale.GERMAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "Zhw%&#,aJdfxg");
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
      Locale locale0 = Locale.GERMANY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) jDOMNodePointer0);
      QName qName0 = new QName("http://www.w3.org/XML/1998/namespace");
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
  public void test55()  throws Throwable  {
      Locale locale0 = Locale.PRC;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot create an attribute for path /@null, operation is not allowed for this type of node
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      Element element0 = new Element("preserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) "<<unknown namespace>>");
      QName qName0 = jDOMNodePointer0.getName();
      jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      assertEquals("preserve", qName0.toString());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      Element element0 = new Element("preserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      Element element0 = new Element("preserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      String string0 = jDOMNodePointer1.asPath();
      assertEquals("/preserve[1]", string0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      CDATA cDATA0 = new CDATA((String) null);
      Locale locale0 = Locale.PRC;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0, (String) null);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/text()[1]", string0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("peserve", "peserve");
      Locale locale0 = Locale.TAIWAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/processing-instruction('peserve')[1]", string0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      Element element0 = new Element("preserve");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      boolean boolean0 = jDOMNodePointer0.equals("<<unknown namespace>>");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Object object0 = new Object();
      Locale locale0 = Locale.TAIWAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(object0, locale0, "space");
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer((Object) null, locale0, "java.util.Locale@0000000009");
      boolean boolean0 = jDOMNodePointer1.equals(jDOMNodePointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(locale0, locale0, "<<unknown namespace>>");
      boolean boolean0 = jDOMNodePointer1.equals(jDOMNodePointer0);
      assertFalse(boolean0);
  }
}