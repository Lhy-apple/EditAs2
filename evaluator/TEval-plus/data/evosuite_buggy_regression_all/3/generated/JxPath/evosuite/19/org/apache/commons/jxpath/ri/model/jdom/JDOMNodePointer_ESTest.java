/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:58:00 GMT 2023
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
import org.jdom.Content;
import org.jdom.Document;
import org.jdom.Element;
import org.jdom.ProcessingInstruction;
import org.jdom.Text;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JDOMNodePointer_ESTest extends JDOMNodePointer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Locale locale0 = Locale.US;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("^.$mOOpTX7*O", locale0, "^.$mOOpTX7*O");
      boolean boolean0 = jDOMNodePointer0.isCollection();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      NodePointer nodePointer0 = jDOMNodePointer0.namespacePointer("Z=D'rit2V^lae2&*jfr");
      assertTrue(nodePointer0.isNode());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = new Document();
      Locale locale0 = Locale.GERMANY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0, "^.$mOOpTX7*O");
      int int0 = jDOMNodePointer0.getLength();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Element element0 = new Element("TjK");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertTrue(boolean0);
      assertEquals("TjK", nodeNameTest0.toString());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      QName qName0 = new QName("jJ", "jJ");
      Element element0 = new Element("jJ", "jJ", "jJ");
      Locale locale0 = Locale.JAPANESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      boolean boolean0 = nodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Element element0 = new Element("og.apache.cmmon.jxpath.ri.model.VaAiablePointr", "og.apache.cmmon.jxpath.ri.model.VaAiablePointr", "og.apache.cmmon.jxpath.ri.model.VaAiablePointr");
      CDATA cDATA0 = new CDATA("og.apache.cmmon.jxpath.ri.model.VaAiablePointr");
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      assertEquals("og.apache.cmmon.jxpath.ri.model.VaAiablePointr", jDOMNodePointer1.getNamespaceURI());
      
      jDOMNodePointer1.asPath();
      String string0 = jDOMNodePointer1.asPath();
      assertEquals("/text()[1]/og.apache.cmmon.jxpath.ri.model.VaAiablePointr:og.apache.cmmon.jxpath.ri.model.VaAiablePointr[1]", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      QName qName0 = new QName("pb");
      Element element0 = new Element("pb");
      Locale locale0 = Locale.ITALIAN;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) nodePointer0);
      NodePointer nodePointer1 = nodePointer0.createAttribute(jXPathContext0, qName0);
      int int0 = nodePointer0.compareChildNodePointers(nodePointer1, nodePointer0);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      QName qName0 = new QName((String) null, "");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(variablePointer0, variablePointer0);
      jDOMNodePointer0.hashCode();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Locale locale0 = Locale.UK;
      CDATA cDATA0 = new CDATA((String) null);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild((JXPathContext) null, qName0, Integer.MIN_VALUE, (Object) locale0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      String string0 = jDOMNodePointer0.getNamespaceURI();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      QName qName0 = new QName("jJ", "jJ");
      Element element0 = new Element("jJ", "jJ", "jJ");
      Locale locale0 = Locale.PRC;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      NodePointer nodePointer1 = nodePointer0.createAttribute((JXPathContext) null, qName0);
      int int0 = nodePointer0.compareChildNodePointers(nodePointer0, nodePointer1);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      Comment comment0 = new Comment("org.apache.commons.jxpath.ri.model.VariablePointer");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "sum");
      String string0 = jDOMNodePointer0.getNamespaceURI("xml");
      assertEquals("http://www.w3.org/XML/1998/namespace", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      Document document0 = new Document();
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0, "http://www.w3.org/2000/xmlns/");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.getNamespaceURI("http://www.w3.org/2000/xmlns/");
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
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      String string0 = jDOMNodePointer0.getNamespaceURI("http://www.w3.org/2000/xmlns/");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      QName qName0 = new QName("DjKJ", "DjKJ");
      Element element0 = new Element("DjKJ", "DjKJ", "GZ");
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      int int0 = nodePointer0.compareChildNodePointers(nodePointer0, nodePointer0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      QName qName0 = new QName("");
      Locale locale0 = Locale.UK;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, "", locale0);
      CDATA cDATA0 = new CDATA("<<unknown namespace>>");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodePointer0, cDATA0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.compareChildNodePointers(nodePointer0, jDOMNodePointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // JXPath internal error: compareChildNodes called for [CDATA: <<unknown namespace>>]
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      Element element0 = new Element("P", "P");
      NodeNameTest nodeNameTest0 = new NodeNameTest((QName) null);
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, nodeNameTest0, locale0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodePointer0, element0);
      jDOMNodePointer0.setValue(nodePointer0);
      int int0 = jDOMNodePointer0.compareChildNodePointers(nodePointer0, jDOMNodePointer0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Element element0 = new Element("i", "i");
      Document document0 = new Document(element0);
      Locale locale0 = new Locale("i", "i", "i");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      QName qName0 = new QName("jJ", "jJ");
      Element element0 = new Element("jJ", "jJ", "jJ");
      Locale locale0 = Locale.JAPANESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      Comment comment0 = new Comment("<<unknown namespace>>");
      nodePointer0.setValue(comment0);
      boolean boolean0 = nodePointer0.isLeaf();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Element element0 = new Element("Dj4KJ", "Dj4KJ", "Dj4KJ");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      QName qName0 = jDOMNodePointer0.getName();
      assertEquals("Dj4KJ:Dj4KJ", qName0.toString());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("K4IdUTq", "K4IdUTq");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      assertEquals("K4IdUTq", qName0.toString());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.VAwiablePointer");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      jDOMNodePointer0.setValue("org.apache.commons.jxpath.ri.model.VAwiablePointer");
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("org.apache.commons.jxpath.ri.model.VAwiablePointer", object0);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Element element0 = new Element("TjK");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      Document document0 = new Document(element0);
      jDOMNodePointer0.setValue(document0);
      jDOMNodePointer0.getValue();
      assertEquals(1, element0.getContentSize());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      QName qName0 = new QName("jJ", "jJ");
      Element element0 = new Element("jJ", "jJ", "jJ");
      Locale locale0 = Locale.JAPANESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      Comment comment0 = new Comment("<<unknown namespace>>");
      nodePointer0.setValue(comment0);
      Object object0 = nodePointer0.getValue();
      assertNotNull(object0);
      assertEquals("", object0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Locale locale0 = Locale.US;
      Comment comment0 = new Comment("OQ'\"?XpHQDU,s");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("OQ'\"?XpHQDU,s", object0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Locale locale0 = Locale.FRANCE;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("K4IdUTq", "K4IdUTq");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("K4IdUTq", object0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CDATA cDATA0 = new CDATA((String) null);
      Locale locale0 = Locale.CHINA;
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
  public void test29()  throws Throwable  {
      Locale locale0 = Locale.UK;
      CDATA cDATA0 = new CDATA((String) null);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      jDOMNodePointer0.setValue("<<unknown namespace>>");
      assertEquals("<<unknown namespace>>", cDATA0.getValue());
      assertEquals("<<unknown namespace>>", cDATA0.getText());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      QName qName0 = new QName("");
      Locale locale0 = Locale.UK;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, "", locale0);
      CDATA cDATA0 = new CDATA("<<unknown namespace>>");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodePointer0, cDATA0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.setValue(nodePointer0);
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
      Element element0 = new Element("Z", "Z", "Z");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      jDOMNodePointer0.setValue(element0);
      assertEquals("Z", element0.getNamespaceURI());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Element element0 = new Element("i");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      CDATA cDATA0 = new CDATA("http://www.w3.org/2000/xmlns/");
      jDOMNodePointer0.setValue(cDATA0);
      assertEquals("http://www.w3.org/2000/xmlns/", cDATA0.getText());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Element element0 = new Element("i");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      jDOMNodePointer0.setValue((Object) null);
      assertEquals(Integer.MIN_VALUE, jDOMNodePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Element element0 = new Element("i");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      LinkedList<String> linkedList0 = new LinkedList<String>();
      jDOMNodePointer0.setValue(linkedList0);
      assertFalse(jDOMNodePointer0.isContainer());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(2);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      QName qName0 = new QName("http://www.w3.org/2000/xmlns/", "<<unknown namespace>>");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "http://www.w3.org/2000/xmlns/");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) jDOMNodePointer0, (NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Element element0 = new Element("DK");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      QName qName0 = new QName("http://www.w3.org/2000/xmlns/", "http://www.w3.org/XML/1998/namespace");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "DK");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Element element0 = new Element("DjKJ");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "DjKJ");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertTrue(boolean0);
      assertEquals("DjKJ", nodeNameTest0.toString());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      QName qName0 = new QName("jJ", "jJ");
      Element element0 = new Element("jJ", "jJ", "jJ");
      Locale locale0 = Locale.PRC;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "jJ");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      QName qName0 = new QName("jJ", "jJ");
      Element element0 = new Element("jJ");
      Locale locale0 = Locale.JAPANESE;
      Object object0 = new Object();
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, object0, locale0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodePointer0, element0);
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "GZ");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Locale locale0 = Locale.KOREAN;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("K4IdUT", "K4IdUT");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("http://www.w3.org/XML/1998/namespace");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Locale locale0 = Locale.KOREA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(3);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Locale locale0 = Locale.US;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(4);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) "DjKJ", (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      CDATA cDATA0 = new CDATA("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(2);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("http://www.w3.org/XML/1998/namespace");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      QName qName0 = new QName("DjKJ", "DjKJ");
      Element element0 = new Element("DjKJ", "DjKJ", "GZ");
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) nodePointer0);
      NodePointer nodePointer1 = nodePointer0.createAttribute(jXPathContext0, qName0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodePointer1, jXPathContext0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/@DjKJ:DjKJ", string0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Element element0 = new Element("org.mpache.commons.jxpNth.ri.model.VAwiablePoZnter", "org.mpache.commons.jxpNth.ri.model.VAwiablePoZnter", "org.mpache.commons.jxpNth.ri.model.VAwiablePoZnter");
      String string0 = JDOMNodePointer.getPrefix(element0);
      assertNotNull(string0);
      assertEquals("org.mpache.commons.jxpNth.ri.model.VAwiablePoZnter", string0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      String string0 = JDOMNodePointer.getPrefix("SMTN,&5CcbZ,Wg!|$ D");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Attribute attribute0 = new Attribute("DjKJ", "DjKJ");
      String string0 = JDOMNodePointer.getPrefix(attribute0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Locale locale0 = Locale.KOREAN;
      String string0 = JDOMNodePointer.getLocalName(locale0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Comment comment0 = new Comment("OQT.'\"?XpHQDU,");
      Locale locale0 = Locale.PRC;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLanguage("OQT.'\"?XpHQDU,");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) null);
      QName qName0 = jDOMNodePointer0.getName();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild(jXPathContext0, qName0, (-337));
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
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, (Object) null);
      QName qName0 = new QName("http://www.w3.org/2000/xmlns/", "http://www.w3.org/XML/1998/namespace");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) qName0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot create an attribute for path /@http://www.w3.org/2000/xmlns/:http://www.w3.org/XML/1998/namespace, operation is not allowed for this type of node
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.VAwiablePointer");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
      QName qName0 = new QName("http://www.w3.org/XML/1998/namespace");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Unknown namespace prefix: http
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      QName qName0 = new QName("jJ", "jJ");
      Element element0 = new Element("jJ", "jJ", "jJ");
      Locale locale0 = Locale.JAPANESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) nodeNameTest0);
      nodePointer0.createAttribute(jXPathContext0, qName0);
      NodePointer nodePointer1 = nodePointer0.createAttribute(jXPathContext0, qName0);
      assertFalse(nodePointer1.isRoot());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      QName qName0 = new QName("pb");
      Element element0 = new Element("pb", "pb", "pb");
      Locale locale0 = Locale.FRENCH;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) nodePointer0);
      nodePointer0.createAttribute(jXPathContext0, qName0);
      NodePointer nodePointer1 = nodePointer0.createAttribute(jXPathContext0, qName0);
      assertEquals(Integer.MIN_VALUE, nodePointer1.getIndex());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Element element0 = new Element("og.apache.cmmon.jxpath.ri.model.VaAiablePointr");
      CDATA cDATA0 = new CDATA("og.apache.cmmon.jxpath.ri.model.VaAiablePointr");
      Locale locale0 = Locale.FRANCE;
      element0.setContent((Content) cDATA0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      jDOMNodePointer0.remove();
      assertFalse(jDOMNodePointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.VAwiablePointer");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, element0);
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
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("rum", locale0, "rum");
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("id('rum')", string0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      QName qName0 = new QName("DjKJ", "DjKJ");
      Element element0 = new Element("DjKJ", "DjKJ", "GZ");
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodePointer0, element0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/DjKJ:DjKJ[1]", string0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      QName qName0 = new QName("DjKJ");
      Element element0 = new Element("DjKJ");
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodePointer0, element0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/DjKJ[1]", string0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("K4IdUTq", "K4IdUTq");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/processing-instruction('K4IdUTq')[1]", string0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Element element0 = new Element("og.apache.cmmon.jxpath.ri.model.VariablePointr", "og.apache.cmmon.jxpath.ri.model.VariablePointr", "og.apache.cmmon.jxpath.ri.model.VariablePointr");
      CDATA cDATA0 = new CDATA("og.apache.cmmon.jxpath.ri.model.VariablePointr");
      Locale locale0 = Locale.FRANCE;
      element0.setContent((Content) cDATA0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/text()[1]", string0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Locale locale0 = Locale.ROOT;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("OQ'\"?XpHQDU,s", locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer((Object) jDOMNodePointer0, locale0);
      boolean boolean0 = jDOMNodePointer1.equals(jDOMNodePointer0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      QName qName0 = new QName("");
      Locale locale0 = Locale.UK;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, "", locale0);
      CDATA cDATA0 = new CDATA((String) null);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodePointer0, cDATA0);
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Object object0 = new Object();
      Locale locale0 = Locale.US;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(object0, locale0);
      boolean boolean0 = jDOMNodePointer0.equals(locale0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Text text0 = new Text("G8%%y");
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(text0, locale0, "G8%%y");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(text0, locale0);
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer1);
      assertTrue(boolean0);
  }
}
