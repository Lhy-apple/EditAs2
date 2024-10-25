/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:54:06 GMT 2023
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
import org.jdom.DocType;
import org.jdom.Document;
import org.jdom.Element;
import org.jdom.Namespace;
import org.jdom.ProcessingInstruction;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JDOMNodePointer_ESTest extends JDOMNodePointer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.isCollection();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Locale locale0 = Locale.ROOT;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, "rg.jdom.Comment@0000000005");
      NodePointer nodePointer0 = jDOMNodePointer0.namespacePointer("rg.jdom.Comment@0000000005");
      assertFalse(nodePointer0.isContainer());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      jDOMNodePointer0.setIndex(18);
      jDOMNodePointer0.isActual();
      assertEquals(1, jDOMNodePointer0.getLength());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Element element0 = new Element("org.jdom.Co.ntentLis");
      Locale locale0 = Locale.forLanguageTag("org.jdom.Co.ntentLis");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("");
      Locale locale0 = Locale.ROOT;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstructionTest0, locale0, "");
      QName qName0 = new QName("http://www.w3.org/XML/1998/namespace", "s-/vy(Ua8W$K!#X3uei");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.childIterator(processingInstructionTest0, false, variablePointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Undefined variable: http://www.w3.org/XML/1998/namespace:s-/vy(Ua8W$K!#X3uei
         //
         verifyException("org.apache.commons.jxpath.ri.model.VariablePointer$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, "UbwW");
      NodeIterator nodeIterator0 = jDOMNodePointer0.namespaceIterator();
      assertEquals(0, nodeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      int int0 = jDOMNodePointer0.compareChildNodePointers(jDOMNodePointer0, jDOMNodePointer0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      Element element0 = new Element("y");
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
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0);
      jDOMNodePointer0.hashCode();
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Locale locale0 = Locale.PRC;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) "http://www.w3.org/XML/1998/namespace");
      QName qName0 = jDOMNodePointer0.getName();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild(jXPathContext0, qName0, 2728, (Object) qName0);
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
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Element element0 = new Element("org.jdom.Co.ntentLis", "org.jdom.Co.ntentLis");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("http://www.w3.org/2000/xmlns/", "org.jdom.Co.ntentLis");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "org.jdom.Co.ntentLis");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Element element0 = new Element("org.jdom.Co.ntentLis", "org.jdom.Co.ntentLis");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("org.jdom.Co.ntentLis");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Element element0 = new Element("eDHd", "eDHd", "eDHd");
      Locale locale0 = Locale.CANADA;
      Document document0 = new Document(element0, (DocType) null, "eDHd");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0, "eDHd");
      String string0 = jDOMNodePointer0.getNamespaceURI("eDHd");
      assertEquals("eDHd", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Element element0 = new Element("eDHd");
      Locale locale0 = Locale.CANADA;
      Document document0 = new Document(element0, (DocType) null, "eDHd");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0, "eDHd");
      String string0 = jDOMNodePointer0.getNamespaceURI("http://www.w3.org/XML/1998/namespace");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Object object0 = new Object();
      Locale locale0 = Locale.ROOT;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(object0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("1HK)");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Namespace namespace0 = Namespace.getNamespace("Ub", "Ub");
      Element element0 = new Element("Ub", namespace0);
      Locale locale0 = Locale.CHINA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("Ub");
      assertEquals("Ub", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Object object0 = new Object();
      Locale locale0 = Locale.ROOT;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(object0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, (Object) null);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.compareChildNodePointers(jDOMNodePointer1, jDOMNodePointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // JXPath internal error: compareChildNodes called for java.lang.Object@78d0c34a
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Element element0 = new Element("org.jdom.Co.ntentLis", "org.jdom.Co.ntentLis");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue("org.jdom.Co.ntentLis");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(Integer.MIN_VALUE, locale0);
      int int0 = jDOMNodePointer0.compareChildNodePointers(jDOMNodePointer0, jDOMNodePointer1);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Element element0 = new Element("eDHd", "eDHd", "eDHd");
      Locale locale0 = Locale.CANADA;
      Document document0 = new Document(element0, (DocType) null, "eDHd");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0, "eDHd");
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Element element0 = new Element("org.jdom.Co.ntentLis", "org.jdom.Co.ntentLis");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Element element0 = new Element("eDHd", "eDHd", "eDHd");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      Comment comment0 = new Comment("java.util.Locale@0000000005");
      jDOMNodePointer0.setValue(comment0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, "UbwW");
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Namespace namespace0 = Namespace.getNamespace("Ub", "Ub");
      Element element0 = new Element("Ub", namespace0);
      Locale locale0 = Locale.JAPANESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      assertEquals("Ub", qName0.getPrefix());
      assertEquals("Ub", qName0.getName());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("org.apache.omons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.omons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.PRC;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      assertEquals("org.apache.omons.jxpath.ri.model.jdom.JDOMNodePointer", qName0.getName());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("org.apache.omons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.omons.jxpath.ri.model.jdom.JDOMNodePointer");
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("org.apache.omons.jxpath.ri.model.jdom.JDOMNodePointer", object0);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Locale locale0 = Locale.TAIWAN;
      Comment comment0 = new Comment("org.jdom.Comment@000000001<");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "org.jdom.Comment@000000001<");
      Object object0 = jDOMNodePointer0.getValue();
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CDATA cDATA0 = new CDATA("G");
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("G", object0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CDATA cDATA0 = new CDATA("k");
      Locale locale0 = Locale.CHINA;
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
      CDATA cDATA0 = new CDATA("k");
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      jDOMNodePointer0.setValue("k");
      assertEquals("k", cDATA0.getText());
      assertEquals("k", cDATA0.getValue());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CDATA cDATA0 = new CDATA("");
      Locale locale0 = Locale.KOREA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0, "Factory could not create a child node for path: ");
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
  public void test32()  throws Throwable  {
      Element element0 = new Element("org.jdom.Co.ntentLis", "org.jdom.Co.ntentLis");
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue(element0);
      assertTrue(jDOMNodePointer0.isNode());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      Namespace namespace0 = Namespace.getNamespace("org6jdom.CMmment@0000C00005");
      Element element0 = new Element("U2bwW", namespace0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "org6jdom.CMmment@0000C00005");
      Document document0 = new Document(element0);
      jDOMNodePointer0.setValue(document0);
      assertEquals(1, element0.getContentSize());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Element element0 = new Element("org.jdom.Co.ntentLis", "org.jdom.Co.ntentLis");
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      CDATA cDATA0 = new CDATA("java.util.Locale@0000000005");
      jDOMNodePointer0.setValue(cDATA0);
      assertTrue(jDOMNodePointer0.isNode());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Namespace namespace0 = Namespace.getNamespace("UbwW");
      Element element0 = new Element("UbwW", namespace0);
      Locale locale0 = Locale.JAPANESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("UbwW", "UbwW");
      jDOMNodePointer0.setValue(processingInstruction0);
      assertEquals(1, jDOMNodePointer0.getLength());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Element element0 = new Element("org.jdom.ContentList", "org.jdom.ContentList", "org.jdom.ContentList");
      Locale locale0 = Locale.KOREA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue((Object) null);
      assertEquals(1, jDOMNodePointer0.getLength());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Namespace namespace0 = Namespace.getNamespace("Ub", "Ub");
      Element element0 = new Element("Ub", namespace0);
      Locale locale0 = Locale.JAPANESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue(jDOMNodePointer0);
      assertEquals(Integer.MIN_VALUE, jDOMNodePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Element element0 = new Element("org.jdom.Co.ntentLis", "org.jdom.Co.ntentLis");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(4);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Element element0 = new Element("org.jdom.Co.ntentLis", "org.jdom.Co.ntentLis", "org.jdom.Co.ntentLis");
      Locale locale0 = Locale.JAPANESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("org.jdom.Co.ntentLis", "<<unknown namespace>>");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("<<unknown namespace>>");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      Locale locale0 = Locale.US;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodeTypeTest0, locale0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Locale locale0 = Locale.TAIWAN;
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(2);
      CDATA cDATA0 = new CDATA("org.apache.commons.jxpath.ri.compiler.NodeTypeTest@0000000007");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(3);
      Locale locale0 = Locale.US;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(nodeTypeTest0, locale0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(2);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("http://www.w3.org/2000/xmlns/");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Element element0 = new Element("org.jdom.Co.ntentLis", "org.jdom.Co.ntentLis");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("org.jdom.Co.ntentLis");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      Element element0 = new Element("deDHd");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "Unknown namespace prefix: ");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertEquals("deDHd", qName0.getName());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Namespace namespace0 = Namespace.getNamespace("org.jdom.Comment@0000000005");
      Element element0 = new Element("UbwW", namespace0);
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "DMPx=?2&|L");
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "http://www.w3.org/2000/xmlns/");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
      assertEquals("UbwW", qName0.getName());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Attribute attribute0 = new Attribute("org.jdom.ContentList", "org.jdom.ContentList");
      String string0 = JDOMNodePointer.getPrefix(attribute0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Element element0 = new Element("org.jdom.ContentList", "org.jdom.ContentList", "org.jdom.ContentList");
      String string0 = JDOMNodePointer.getPrefix(element0);
      assertNotNull(string0);
      assertEquals("org.jdom.ContentList", string0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Element element0 = new Element("org.jdom.ContentList", "org.jdom.ContentList");
      String string0 = JDOMNodePointer.getPrefix(element0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Locale locale0 = Locale.ITALIAN;
      String string0 = JDOMNodePointer.getPrefix(locale0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Attribute attribute0 = new Attribute("org.jdom.ContentList", "org.jdom.ContentList");
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      attribute0.setNamespace(namespace0);
      String string0 = JDOMNodePointer.getPrefix(attribute0);
      assertEquals("xml", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Namespace namespace0 = Namespace.getNamespace("UgwW", "=@WXiC,,~_p3/{LXz");
      Attribute attribute0 = new Attribute("UgwW", "UgwW", namespace0);
      String string0 = JDOMNodePointer.getLocalName(attribute0);
      assertEquals("UgwW", string0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      String string0 = JDOMNodePointer.getLocalName("pwj(_dfijl");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLanguage("U2bwW");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      Namespace namespace0 = Namespace.getNamespace("org6jdom.CMmment@0000C00005");
      Element element0 = new Element("U2bwW", namespace0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "org6jdom.CMmment@0000C00005");
      boolean boolean0 = jDOMNodePointer0.isLanguage("U2bwW");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("org.apache.omons.jxpath.ri.modl.jdom.JDOMNodePointer", "org.apache.omons.jxpath.ri.modl.jdom.JDOMNodePointer");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLanguage("org.apache.omons.jxpath.ri.modl.jdom.JDOMNodePointer");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      Comment comment0 = new Comment("org.jdom.Commenta000000001<");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0, "org.jdom.Commenta000000001<");
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
  public void test63()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) "http://www.w3.org/XML/1998/namespace");
      QName qName0 = jDOMNodePointer0.getName();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild(jXPathContext0, qName0, Integer.MIN_VALUE, (Object) locale0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Factory is not set on the JXPathContext - cannot create path: 
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Namespace namespace0 = Namespace.getNamespace("org.jdom.Comment@0000000005");
      Element element0 = new Element("UbwW", namespace0);
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "DMPx=?2&|L");
      QName qName0 = jDOMNodePointer0.getName();
      Object object0 = new Object();
      JXPathContext jXPathContext0 = JXPathContext.newContext((JXPathContext) null, object0);
      jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      NodePointer nodePointer0 = jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      assertFalse(nodePointer0.isCollection());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      QName qName0 = new QName("<<unknown namespace>>", (String) null);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot create an attribute for path /@<<unknown namespace>>:null, operation is not allowed for this type of node
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Namespace namespace0 = Namespace.getNamespace("org.jdom.Comment@0000000005");
      Element element0 = new Element("UbW", namespace0);
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "DMPx=?2&|L");
      QName qName0 = new QName("xml", "UbW");
      jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
      NodePointer nodePointer0 = jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
      assertTrue(nodePointer0.isNode());
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Namespace namespace0 = Namespace.getNamespace("org.dom.Comment@00000005");
      Element element0 = new Element("UbwW", namespace0);
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "DMPx=?2&|L");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) namespace0);
      QName qName0 = new QName("http://www.w3.org/XML/1998/namespace", "UbwW");
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
  public void test68()  throws Throwable  {
      Locale locale0 = Locale.GERMAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0, "Ypzq\"]<Z).:N],>");
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("id('Ypzq&quot;]<Z).:N],>')", string0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Element element0 = new Element("eDHd", "eDHd", "eDHd");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      CDATA cDATA0 = new CDATA("G");
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/text()[1]", string0);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", "org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((NodePointer) null, processingInstruction0);
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("/processing-instruction('org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer')[1]", string0);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Locale locale0 = Locale.FRANCE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0, "/_I(uXL<H'w@H*Q'7-");
      String string0 = jDOMNodePointer0.asPath();
      assertEquals("id('/_I(uXL<H&apos;w@H*Q&apos;7-')", string0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Object object0 = new Object();
      Locale locale0 = Locale.ROOT;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(object0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, (Object) null);
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Object object0 = new Object();
      Locale locale0 = Locale.ROOT;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(object0, locale0);
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Locale locale0 = Locale.GERMANY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, (Locale) null);
      boolean boolean0 = jDOMNodePointer0.equals(locale0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer((NodePointer) jDOMNodePointer0, (Object) locale0);
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer1);
      assertTrue(boolean0);
  }
}
