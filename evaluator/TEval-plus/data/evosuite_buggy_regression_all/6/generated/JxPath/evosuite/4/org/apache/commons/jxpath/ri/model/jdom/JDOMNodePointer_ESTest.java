/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:40:35 GMT 2023
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
import org.apache.commons.jxpath.ri.model.NodeIterator;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.testdata.EvoSuiteFile;
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
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.isCollection();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, "r}g\"X:9o");
      NodePointer nodePointer0 = jDOMNodePointer0.namespacePointer("http://www.w3.org/2000/xmlns/");
      assertTrue(nodePointer0.isNode());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0, "-%;++xn=C7nLWwu");
      Namespace namespace0 = Namespace.getNamespace("<<unknown namespace>>");
      Element element0 = new Element("Y", namespace0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      NamespaceResolver namespaceResolver0 = new NamespaceResolver((NamespaceResolver) null);
      namespaceResolver0.setNamespaceContextPointer(jDOMNodePointer1);
      jDOMNodePointer1.setNamespaceResolver(namespaceResolver0);
      String string0 = jDOMNodePointer1.asPath();
      assertEquals("id('-%;++xn=C7nLWwu')/node()[1]", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Locale locale0 = Locale.CANADA;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("goL", "goL");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      int int0 = jDOMNodePointer0.getLength();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Locale locale0 = Locale.KOREAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("http://www.w3.org/2000/xmlns/");
      NodeIterator nodeIterator0 = jDOMNodePointer0.childIterator(processingInstructionTest0, false, jDOMNodePointer0);
      assertEquals(0, nodeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      Element element0 = new Element("YL", namespace0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "[ProcessingInstruction: ");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) "YL");
      QName qName0 = jDOMNodePointer0.getName();
      jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      assertEquals("xml:YL", qName0.toString());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Comment comment0 = new Comment("");
      Locale locale0 = Locale.CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "");
      jDOMNodePointer0.hashCode();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild((JXPathContext) null, qName0, Integer.MIN_VALUE, (Object) locale0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Element element0 = new Element("E2");
      Locale locale0 = Locale.forLanguageTag("E2");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "E2");
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "E2");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) element0, (NodeTest) nodeNameTest0);
      assertFalse(boolean0);
      assertEquals("E2", nodeNameTest0.toString());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Element element0 = new Element("orjdom.IlTegjlNamExcet.of", "orjdom.IlTegjlNamExcet.of", "orjdom.IlTegjlNamExcet.of");
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("orjdom.IlTegjlNamExcet.of");
      assertEquals("orjdom.IlTegjlNamExcet.of", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = new Document();
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.dynamic.DynamicPointerFactory");
      document0.setRootElement(element0);
      Locale locale0 = Locale.TAIWAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("http://www.w3.org/2000/xmlns/");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document();
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.dynamic.DynamicPointerFactory", "org.apache.commons.jxpath.ri.model.dynamic.DynamicPointerFactory", "org.apache.commons.jxpath.ri.model.dynamic.DynamicPointerFactory");
      Document document1 = document0.setRootElement(element0);
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document1, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("org.apache.commons.jxpath.ri.model.dynamic.DynamicPointerFactory");
      assertEquals("org.apache.commons.jxpath.ri.model.dynamic.DynamicPointerFactory", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Locale locale0 = new Locale("[Processingnstrction:j", "[Processingnstrction:j", "[Processingnstrction:j");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, "[Processingnstrction:j");
      String string0 = jDOMNodePointer0.getNamespaceURI("[Processingnstrction:j");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Element element0 = new Element("orm.jdm.IlHegjlNameException", "orm.jdm.IlHegjlNameException");
      Locale locale0 = new Locale("orm.jdm.IlHegjlNameException");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "orm.jdm.IlHegjlNameException");
      String string0 = jDOMNodePointer0.getNamespaceURI("orm.jdm.IlHegjlNameException");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      int int0 = jDOMNodePointer0.compareChildNodePointers(jDOMNodePointer0, jDOMNodePointer0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("", locale0, "");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer((Object) jDOMNodePointer0, locale0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.compareChildNodePointers(jDOMNodePointer1, jDOMNodePointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // JXPath internal error: compareChildNodes called for 
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      Element element0 = new Element("orjdom.IlTegjlNamExcet.on", "orjdom.IlTegjlNamExcet.on");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "orjdom.IlTegjlNamExcet.on");
      QName qName0 = jDOMNodePointer0.getName();
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, "orjdom.IlTegjlNamExcet.on", locale0);
      jDOMNodePointer0.setValue("orjdom.IlTegjlNamExcet.on");
      int int0 = jDOMNodePointer0.compareChildNodePointers(nodePointer0, jDOMNodePointer0);
      assertEquals("orjdom.IlTegjlNamExcet.on", qName0.toString());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      Element element0 = new Element("h", "h");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "h");
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      Element element0 = new Element("h", "h");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "h");
      jDOMNodePointer0.setValue("h");
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Locale locale0 = Locale.US;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document();
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.dynamic.DynamicPointerFactory", "org.apache.commons.jxpath.ri.model.dynamic.DynamicPointerFactory", "org.apache.commons.jxpath.ri.model.dynamic.DynamicPointerFactory");
      Document document1 = document0.setRootElement(element0);
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document1, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Locale locale0 = Locale.CANADA;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("RA7", "RA7");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      assertEquals("RA7", qName0.getName());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      Element element0 = new Element("t-", "t-");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "?qG2");
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("", object0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Locale locale0 = Locale.GERMANY;
      Comment comment0 = new Comment("");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("", object0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Locale locale0 = Locale.ITALIAN;
      CDATA cDATA0 = new CDATA("+");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("+", object0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Locale locale0 = Locale.CANADA;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("RL", "RL");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("RL", object0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Locale locale0 = Locale.ROOT;
      CDATA cDATA0 = new CDATA("org.jdom.IlHegalNameException");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0, "org.jdom.IlHegalNameException");
      LinkedList<String> linkedList0 = new LinkedList<String>();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.setValue(linkedList0);
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
      Locale locale0 = Locale.US;
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
  public void test31()  throws Throwable  {
      CDATA cDATA0 = new CDATA("/processing-instruction('");
      Locale locale0 = Locale.US;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      jDOMNodePointer0.setValue(locale0);
      assertEquals(Integer.MIN_VALUE, NodePointer.WHOLE_COLLECTION);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      Element element0 = new Element("h", namespace0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "h");
      jDOMNodePointer0.setValue(element0);
      assertEquals("h", element0.getName());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Locale locale0 = Locale.GERMAN;
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      Element element0 = new Element("h", namespace0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "h");
      CDATA cDATA0 = new CDATA("Et<5l~Ue3x&iE");
      jDOMNodePointer0.setValue(cDATA0);
      assertEquals(1, jDOMNodePointer0.getLength());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      Element element0 = new Element("h", "h");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("h", "org.apache.commons.jxpath.ri.compiler.ProcessingInstructionTest@0000000007");
      jDOMNodePointer0.setValue(processingInstruction0);
      assertTrue(jDOMNodePointer0.isRoot());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      Element element0 = new Element("YL");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "[Processingnstrction:j");
      Object object0 = new Object();
      JXPathContext jXPathContext0 = JXPathContext.newContext(object0);
      Comment comment0 = new Comment(",^w&vKmDkEuG");
      NodePointer nodePointer0 = jDOMNodePointer0.createPath(jXPathContext0, (Object) comment0);
      assertFalse(nodePointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Element element0 = new Element("orm.jdom.IlHegjlNameExcept.on");
      Locale locale0 = Locale.CHINA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "orm.jdom.IlHegjlNameExcept.on");
      jDOMNodePointer0.setValue((Object) null);
      assertTrue(jDOMNodePointer0.isRoot());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Element element0 = new Element("orm.jdm.IlHeglNameException", "orm.jdm.IlHeglNameException");
      Locale locale0 = new Locale("orm.jdm.IlHeglNameException");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "orm.jdm.IlHeglNameException");
      jDOMNodePointer0.setValue("");
      assertTrue(jDOMNodePointer0.isRoot());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Element element0 = new Element("orm.jdm.IlHeglNameException", "orm.jdm.IlHeglNameException");
      Locale locale0 = Locale.GERMAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "orm.jdm.IlHeglNameException");
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      Document document0 = new Document(linkedList0);
      Document document1 = document0.setContent((Content) element0);
      jDOMNodePointer0.setValue(document1);
      assertEquals(1, element0.getContentSize());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) "orjdm.IleglNameExcepon", (NodeTest) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      Element element0 = new Element("h");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "h");
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) element0, (NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Element element0 = new Element("orm.jdom.IlHegjlNameException", "orm.jdom.IlHegjlNameException", "orm.jdom.IlHegjlNameException");
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, ": ");
      QName qName0 = new QName("<<unknown namespace>>");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Locale locale0 = Locale.CANADA;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("RL", "RL");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("<<unknown namespace>>");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0, "[ProcssingInstructio-: ");
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(4);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) null, (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(3);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) nodeTypeTest0, (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) null, (NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Document document0 = new Document();
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) document0, (NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("http://www.w3.org/2000/xmlns/");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Element element0 = new Element("orm.jdm.IlHeglNameException", "orm.jdm.IlHeglNameException");
      Locale locale0 = new Locale("orm.jdm.IlHeglNameException");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "orm.jdm.IlHeglNameException");
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) jDOMNodePointer0, (Object) element0, (NodeTest) nodeNameTest0);
      assertFalse(boolean0);
      assertEquals("orm.jdm.IlHeglNameException", nodeNameTest0.toString());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      Element element0 = new Element("orjdom.IlTegjlNamExcet.on", "orjdom.IlTegjlNamExcet.on");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "orjdom.IlTegjlNamExcet.on");
      String string0 = JDOMNodePointer.getPrefix(jDOMNodePointer0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Element element0 = new Element("org.jdom.IlHegalNameException");
      String string0 = JDOMNodePointer.getPrefix(element0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Element element0 = new Element("org.jdom.IlHegalNameException", "org.jdom.IlHegalNameException", "org.jdom.IlHegalNameException");
      String string0 = JDOMNodePointer.getPrefix(element0);
      assertEquals("org.jdom.IlHegalNameException", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      Attribute attribute0 = new Attribute("W", "W", namespace0);
      String string0 = JDOMNodePointer.getPrefix(attribute0);
      assertNotNull(string0);
      assertEquals("xml", string0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Attribute attribute0 = new Attribute("W", "W");
      String string0 = JDOMNodePointer.getPrefix(attribute0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Attribute attribute0 = new Attribute("YL", "YL", 1);
      String string0 = JDOMNodePointer.getLocalName(attribute0);
      assertEquals("YL", string0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      String string0 = JDOMNodePointer.getLocalName("YL");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      Element element0 = new Element("h", "h", "h");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "h");
      boolean boolean0 = jDOMNodePointer0.isLanguage("http://www.w3.org/2000/xmlns/");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Text text0 = new Text("[Comment: ");
      Locale locale0 = Locale.ROOT;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(text0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLanguage("[Comment: ");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
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
  public void test60()  throws Throwable  {
      Locale locale0 = Locale.CANADA;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("RL", "RL");
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
      Locale locale0 = Locale.ITALIAN;
      Comment comment0 = new Comment("+");
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
  public void test62()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) locale0);
      QName qName0 = jDOMNodePointer0.getName();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild(jXPathContext0, qName0, 2013);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Factory is not set on the JXPathContext - cannot create path: 
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Locale locale0 = Locale.GERMANY;
      Comment comment0 = new Comment("");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0);
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
  public void test64()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      Element element0 = new Element("h", "h");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "h");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) jDOMNodePointer0);
      QName qName0 = jDOMNodePointer0.getName();
      NodePointer nodePointer0 = jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      assertFalse(nodePointer0.isAttribute());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Locale locale0 = Locale.KOREAN;
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      Element element0 = new Element("YL", namespace0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "[Processingnstrction:j");
      CDATA cDATA0 = new CDATA("http://www.w3.org/2000/xmlns/");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) cDATA0);
      QName qName0 = new QName("YL", (String) null);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Unknown namespace prefix: YL
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      Element element0 = new Element("Fm", namespace0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "[ProcessingInstruction: ");
      QName qName0 = new QName("Fm");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) "Fm");
      NodePointer nodePointer0 = jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
      NodePointer nodePointer1 = jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      assertNotSame(nodePointer1, nodePointer0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Element element0 = new Element("org.jdom.IlHegalNameException", "org.jdom.IlHegalNameException", "org.jdom.IlHegalNameException");
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Locale locale0 = Locale.US;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      Element element0 = new Element("d");
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
  public void test69()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0, "[ProcssingInstructio-: ");
      Element element0 = new Element("YL");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      NamespaceResolver namespaceResolver0 = new NamespaceResolver((NamespaceResolver) null);
      jDOMNodePointer1.setNamespaceResolver(namespaceResolver0);
      String string0 = jDOMNodePointer1.asPath();
      assertEquals("id('[ProcssingInstructio-: ')/YL[1]", string0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Text text0 = new Text("org.jdom.IlHegalNameException");
      Locale locale0 = Locale.ROOT;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(text0, locale0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/text()[1]", string0);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Locale locale0 = Locale.CANADA;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("RA7", "RA7");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/processing-instruction('RA7')[1]", string0);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Locale locale0 = Locale.GERMANY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer("')", locale0, "')");
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("id('&apos;)')", string0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      CDATA cDATA0 = new CDATA("");
      Locale locale0 = Locale.CHINA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0, "h(\")H");
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("id('h(&quot;)H')", string0);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Comment comment0 = new Comment("");
      Locale locale0 = Locale.CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "");
      CDATA cDATA0 = new CDATA("http://www.w3.org/2000/xmlns/");
      boolean boolean0 = jDOMNodePointer0.equals(cDATA0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Locale locale0 = Locale.CHINA;
      Comment comment0 = new Comment("']");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0);
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Locale locale0 = Locale.US;
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(namespace0, locale0);
      Comment comment0 = new Comment("http://www.w3.org/2000/xmlns/");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(comment0, (Locale) null);
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      Comment comment0 = new Comment("http://www.w3.org/2000/xmlns/");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, (Locale) null);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(comment0, (Locale) null, "Mb1Z<T0|Xog=>WF&K'");
      boolean boolean0 = jDOMNodePointer1.equals(jDOMNodePointer0);
      assertTrue(boolean0);
  }
}
